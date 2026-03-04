#!/usr/bin/env python3
"""Prototype CNS 2026 abstract pipeline for parsing, embedding, and visualization."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import matplotlib
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

matplotlib.use("Agg")

import hdbscan
from nilearn import plotting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PDF_PATH = PROJECT_ROOT / "CNS_2026_Abstracts.pdf"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache" / "cns_2026_prototype"
SCRIPT_LOG = CACHE_DIR / "pipeline.log"
PARSED_CSV = CACHE_DIR / "parsed_abstracts.csv"
SPECTER_MODEL_ID = "allenai/specter2_aug2023refresh_base"
SAMPLE_SEED = 42

try:
    from neurovlm.data import load_masker  # noqa: E402
    from neurovlm.models import load_model as load_neurovlm_model  # noqa: E402
except ModuleNotFoundError:
    for neurovlm_src in (PROJECT_ROOT / "neurovlm" / "src", WORKSPACE_ROOT / "neurovlm" / "src"):
        if neurovlm_src.exists():
            sys.path.insert(0, str(neurovlm_src))
            break
    else:
        raise FileNotFoundError(
            "Could not import the installed `neurovlm` package and could not locate a source checkout. "
            "Expected either an environment install or one of "
            f"{PROJECT_ROOT / 'neurovlm' / 'src'} / {WORKSPACE_ROOT / 'neurovlm' / 'src'}."
        )

    from neurovlm.data import load_masker  # noqa: E402
    from neurovlm.models import load_model as load_neurovlm_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Path to the CNS abstracts PDF.")
    parser.add_argument("--sample-size", type=int, default=100, help="How many abstracts to run in the prototype.")
    parser.add_argument(
        "--sample-strategy",
        choices=("head", "random"),
        default="head",
        help="How to choose the prototype subset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for SPECTER inference. Lower this if RAM is tight.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=180,
        help="Approximate word budget per abstract chunk before averaging embeddings.",
    )
    parser.add_argument(
        "--brainmap-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="Parallel workers used only for rendering PNG brainmap previews.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached artifacts.")
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cns-prototype")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(SCRIPT_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def timed(stage_name: str, logger: logging.Logger):
    class _Timer:
        def __enter__(self_inner):
            self_inner.started = time.perf_counter()
            logger.info("Starting %s", stage_name)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            self_inner.elapsed = time.perf_counter() - self_inner.started
            if exc:
                logger.exception("%s failed after %.2fs", stage_name, self_inner.elapsed)
                return False
            logger.info("Finished %s in %.2fs", stage_name, self_inner.elapsed)
            return False

    return _Timer()


def run_command(command: list[str], logger: logging.Logger) -> str:
    logger.info("Running command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr.strip():
        logger.info("Command stderr: %s", completed.stderr.strip())
    return completed.stdout


def pdf_dimensions(pdf_path: Path, logger: logging.Logger) -> tuple[int, int]:
    info = run_command(["pdfinfo", str(pdf_path)], logger)
    match = re.search(r"Page size:\s+(\d+)\s+x\s+(\d+)", info)
    if not match:
        raise RuntimeError("Could not determine page size from pdfinfo output.")
    return int(match.group(1)), int(match.group(2))


def extract_pdf_column_text(pdf_path: Path, x: int, y: int, width: int, height: int, logger: logging.Logger) -> list[str]:
    output = run_command(
        [
            "pdftotext",
            "-layout",
            "-x",
            str(x),
            "-y",
            str(y),
            "-W",
            str(width),
            "-H",
            str(height),
            str(pdf_path),
            "-",
        ],
        logger,
    )
    pages = output.split("\f")
    if pages and pages[-1] == "":
        pages = pages[:-1]
    return pages


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("\t", " ")
    return re.sub(r"\s+$", "", line)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("Poster Session "):
        return True
    if re.fullmatch(r"(Saturday|Sunday|Monday), .+", stripped):
        return True
    if re.fullmatch(r"Page \d+", stripped):
        return True
    if stripped in {"Pa", "Pag", "Page", "e1", "e2", "e3"}:
        return True
    if re.fullmatch(r"e\d+", stripped):
        return True
    return False


def looks_like_author_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "@" in stripped or ";" in stripped:
        return True
    if re.match(r"^\d+[A-Za-z(]", stripped):
        return True
    if re.search(r"\b[A-Z][A-Za-z'.-]+[0-9]+", stripped):
        return True
    if stripped.count(",") >= 2 and re.search(r"\d", stripped):
        return True
    return False


def looks_like_affiliation_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^\d+[A-Za-z(]", stripped):
        return True
    if re.search(r"\b(University|Institute|Hospital|Department|Centre|Center|College|Faculty|Laboratory|Lab)\b", stripped) and "." not in stripped:
        return True
    return False


def collapse_paragraph_lines(lines: Iterable[str]) -> str:
    text = " ".join(part.strip() for part in lines if part.strip())
    text = text.replace(" - ", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=\w)- (?=\w)", "", text)
    return text.strip()


def looks_like_prose(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) < 5:
        return False
    stopwords = {"the", "and", "of", "to", "in", "with", "for", "that", "this", "we", "is", "was", "were"}
    return sum(word.lower() in stopwords for word in words) >= 1


def split_affiliation_prefix(line: str) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<prefix>\d+\s+.+?\b(?:University|Institute|Hospital|College|Centre|Center|Department|Faculty|Laboratory|Lab)\b)(?:,?)(?:\s+)(?P<rest>.+)$",
        line.strip(),
    )
    if not match:
        return None
    rest = match.group("rest").strip()
    if not looks_like_prose(rest):
        return None
    return match.group("prefix").strip(), rest


def clean_author_names(author_lines: list[str]) -> str:
    raw_block = collapse_paragraph_lines(author_lines)
    if not raw_block:
        return ""
    authors = raw_block.split(";", 1)[0]
    authors = re.sub(r"\([^)]*@[^)]*\)", "", authors)
    authors = re.sub(r"(?<=[A-Za-z\)])\d+(?:,\d+)*", "", authors)
    authors = re.sub(r"\s+,", ",", authors)
    authors = re.sub(r"\s{2,}", " ", authors)
    return authors.strip(" ,;")


def parse_record(raw_lines: list[str]) -> dict[str, str] | None:
    if not raw_lines:
        return None
    first = raw_lines[0].strip()
    match = re.match(r"^([A-Z]{1,3}\d+)\s*-\s*(.+)$", first)
    if not match:
        return None

    poster_number = match.group(1)
    title_lines = [match.group(2).strip()]
    body_lines = [line for line in raw_lines[1:] if line.strip()]

    idx = 0
    while idx < len(body_lines) and not looks_like_author_line(body_lines[idx]):
        title_lines.append(body_lines[idx].strip())
        idx += 1

    author_lines: list[str] = []
    while idx < len(body_lines):
        line = body_lines[idx]
        if line.startswith("Topic Area:"):
            break
        prefixed_affiliation = split_affiliation_prefix(line)
        if prefixed_affiliation and author_lines:
            prefix, remainder = prefixed_affiliation
            author_lines.append(prefix)
            body_lines[idx] = remainder
            break
        if author_lines and not looks_like_author_line(line) and not looks_like_affiliation_line(line):
            if author_lines[-1].rstrip().endswith((" of", ",", "&")) or len(line.split()) <= 4:
                author_lines.append(line)
                idx += 1
                continue
            break
        if not author_lines and not looks_like_author_line(line) and not looks_like_affiliation_line(line):
            break
        author_lines.append(line)
        idx += 1

    abstract_lines: list[str] = []
    topic_area = ""
    while idx < len(body_lines):
        line = body_lines[idx].strip()
        idx += 1
        if not line:
            continue
        if line.startswith("Topic Area:"):
            topic_area = line.split(":", 1)[1].strip()
            break
        abstract_lines.append(line)

    if abstract_lines and author_lines and author_lines[-1].rstrip().endswith(" of"):
        parts = abstract_lines[0].split(maxsplit=1)
        if len(parts) == 2 and parts[0][0].isupper() and looks_like_prose(parts[1]):
            author_lines.append(parts[0])
            abstract_lines[0] = parts[1]

    record = {
        "poster_number": poster_number,
        "title": collapse_paragraph_lines(title_lines),
        "authors": clean_author_names(author_lines),
        "abstract": collapse_paragraph_lines(abstract_lines),
        "topic_area": topic_area,
    }
    if not record["abstract"]:
        return None
    return record


def parse_pdf_abstracts(pdf_path: Path, logger: logging.Logger) -> pd.DataFrame:
    width, height = pdf_dimensions(pdf_path, logger)
    gutter = 12
    left_width = (width // 2) - (gutter // 2)
    right_x = (width // 2) + (gutter // 2)
    right_width = width - right_x
    logger.info(
        "Using PDF crop layout: width=%s height=%s left_width=%s right_x=%s right_width=%s",
        width,
        height,
        left_width,
        right_x,
        right_width,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(extract_pdf_column_text, pdf_path, 0, 0, left_width, height, logger)
        right_future = pool.submit(extract_pdf_column_text, pdf_path, right_x, 0, right_width, height, logger)
        left_pages = left_future.result()
        right_pages = right_future.result()

    if len(left_pages) != len(right_pages):
        raise RuntimeError("Left/right column extraction returned different page counts.")

    poster_start = re.compile(r"^\s*([A-Z]{1,3}\d+)\s*-\s+")
    records: list[dict[str, str]] = []
    current_lines: list[str] = []

    for left_page, right_page in zip(left_pages, right_pages):
        combined_lines: list[str] = []
        for page_text in (left_page, right_page):
            for raw_line in page_text.splitlines():
                line = normalize_line(raw_line)
                if is_noise_line(line):
                    continue
                combined_lines.append(line)

        for line in combined_lines:
            if poster_start.match(line):
                record = parse_record(current_lines)
                if record:
                    records.append(record)
                current_lines = [line]
                continue
            if current_lines:
                current_lines.append(line)

    final_record = parse_record(current_lines)
    if final_record:
        records.append(final_record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No abstracts were parsed from the PDF.")
    frame = frame.drop_duplicates(subset=["poster_number"], keep="first").reset_index(drop=True)
    return frame


def sentence_chunks(text: str, max_words: int) -> list[str]:
    if not text.strip():
        return [""]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current and current_words + len(words) > max_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_words = len(words)
            continue
        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks or [text.strip()]


def build_chunk_manifest(sample_df: pd.DataFrame, max_words: int) -> tuple[list[dict[str, str]], list[list[int]]]:
    payloads: list[dict[str, str]] = []
    record_to_chunk_ids: list[list[int]] = []
    for row in sample_df.itertuples(index=False):
        chunk_ids: list[int] = []
        for chunk in sentence_chunks(str(row.abstract), max_words=max_words):
            payloads.append({"title": str(row.title), "abstract": chunk})
            chunk_ids.append(len(payloads) - 1)
        record_to_chunk_ids.append(chunk_ids)
    return payloads, record_to_chunk_ids


def encode_specter_batch(
    tokenizer: Any,
    model: Any,
    payloads: list[dict[str, str]],
    device: torch.device,
) -> torch.Tensor:
    sep = tokenizer.sep_token or " [SEP] "
    texts = [f"{item['title']}{sep}{item['abstract']}" for item in payloads]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].detach().cpu()


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def aggregate_chunk_embeddings(
    chunk_embeddings: np.ndarray,
    record_to_chunk_ids: list[list[int]],
    normalize_output: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for chunk_ids in record_to_chunk_ids:
        vectors = chunk_embeddings[chunk_ids]
        averaged = vectors.mean(axis=0)
        rows.append(averaged.astype(np.float32))
    matrix = np.vstack(rows).astype(np.float32)
    if normalize_output:
        matrix = l2_normalize(matrix.astype(np.float32)).astype(np.float32)
    return matrix


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_cached_embeddings(path: Path, expected_posters: list[str]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    posters = payload["poster_numbers"].astype(str).tolist()
    if posters != expected_posters:
        return None
    return {name: payload[name] for name in payload.files}


def build_embeddings(
    sample_df: pd.DataFrame,
    batch_size: int,
    chunk_words: int,
    cache_path: Path,
    force: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    expected_posters = sample_df["poster_number"].astype(str).tolist()
    cached = None if force else load_cached_embeddings(cache_path, expected_posters)
    if cached is not None:
        logger.info("Loading cached embeddings from %s", cache_path)
        timings = {
            "specter_model_load_s": float(cached["specter_model_load_s"][0]),
            "neurovlm_model_load_s": float(cached["neurovlm_model_load_s"][0]),
            "chunk_inference_s": float(cached["chunk_inference_s"][0]),
        }
        timings["sample_embedding_runtime_s"] = (
            timings["specter_model_load_s"] + timings["neurovlm_model_load_s"] + timings["chunk_inference_s"]
        )
        return cached["specter_embeddings"], cached["neurovlm_embeddings"], timings

    chunk_payloads, record_to_chunk_ids = build_chunk_manifest(sample_df, max_words=chunk_words)
    logger.info("Prepared %s chunks across %s abstracts", len(chunk_payloads), len(sample_df))

    timings: dict[str, float] = {}
    device = torch.device("cpu")

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL_ID)
    specter_model = AutoModel.from_pretrained(SPECTER_MODEL_ID).to(device).eval()
    timings["specter_model_load_s"] = time.perf_counter() - start

    start = time.perf_counter()
    proj_head = load_neurovlm_model("proj_head_text_mse").to(device).eval()
    timings["neurovlm_model_load_s"] = time.perf_counter() - start

    chunk_specter = np.zeros((len(chunk_payloads), 768), dtype=np.float32)
    chunk_neuro = np.zeros((len(chunk_payloads), 384), dtype=np.float32)

    inference_started = time.perf_counter()
    for start_idx in tqdm(range(0, len(chunk_payloads), batch_size), desc="SPECTER batches", unit="batch"):
        batch = chunk_payloads[start_idx : start_idx + batch_size]
        specter_tensor = encode_specter_batch(tokenizer, specter_model, batch, device=device)
        specter_np = specter_tensor.numpy().astype(np.float32)
        normalized_specter = specter_tensor / specter_tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            neuro_tensor = proj_head(normalized_specter.to(device)).detach().cpu()
        end_idx = start_idx + len(batch)
        chunk_specter[start_idx:end_idx] = specter_np
        chunk_neuro[start_idx:end_idx] = neuro_tensor.numpy().astype(np.float32)
    timings["chunk_inference_s"] = time.perf_counter() - inference_started
    timings["sample_embedding_runtime_s"] = (
        timings["specter_model_load_s"] + timings["neurovlm_model_load_s"] + timings["chunk_inference_s"]
    )

    specter_embeddings = aggregate_chunk_embeddings(chunk_specter, record_to_chunk_ids, normalize_output=True)
    neurovlm_embeddings = aggregate_chunk_embeddings(chunk_neuro, record_to_chunk_ids, normalize_output=False)

    save_npz(
        cache_path,
        poster_numbers=np.array(expected_posters, dtype="U32"),
        specter_embeddings=specter_embeddings,
        neurovlm_embeddings=neurovlm_embeddings,
        specter_model_load_s=np.array([timings["specter_model_load_s"]], dtype=np.float32),
        neurovlm_model_load_s=np.array([timings["neurovlm_model_load_s"]], dtype=np.float32),
        chunk_inference_s=np.array([timings["chunk_inference_s"]], dtype=np.float32),
    )
    logger.info("Saved embeddings cache to %s", cache_path)
    return specter_embeddings, neurovlm_embeddings, timings


def decode_brainmaps(
    neurovlm_embeddings: np.ndarray,
    sample_df: pd.DataFrame,
    brainmap_dir: Path,
    force: bool,
    workers: int,
    logger: logging.Logger,
) -> tuple[dict[str, str], dict[str, float]]:
    brainmap_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = brainmap_dir / "render_metrics.json"
    expected_paths = {row.poster_number: brainmap_dir / f"{row.poster_number}.png" for row in sample_df.itertuples(index=False)}
    if not force and all(path.exists() for path in expected_paths.values()):
        logger.info("Using cached brainmap PNGs from %s", brainmap_dir)
        sample_runtime = 0.0
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            sample_runtime = float(payload.get("sample_brainmap_runtime_s", 0.0))
        return {poster: encode_png(path) for poster, path in expected_paths.items()}, {
            "sample_brainmap_runtime_s": sample_runtime
        }

    device = torch.device("cpu")
    autoencoder = load_neurovlm_model("autoencoder").to(device).eval()
    masker = load_masker()

    latent_tensor = torch.tensor(neurovlm_embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        flatmaps = torch.sigmoid(autoencoder.decoder(latent_tensor)).detach().cpu().numpy().astype(np.float32)

    render_jobs: list[tuple[Path, np.ndarray]] = []
    for row, flatmap in zip(sample_df.itertuples(index=False), flatmaps, strict=True):
        render_jobs.append((expected_paths[row.poster_number], flatmap))

    render_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(render_brainmap_png, out_path, flatmap, masker, logger): out_path.name
            for out_path, flatmap in render_jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Brainmap PNGs", unit="img"):
            future.result()

    sample_runtime = time.perf_counter() - render_started
    metrics_path.write_text(
        json.dumps({"sample_brainmap_runtime_s": sample_runtime}, indent=2),
        encoding="utf-8",
    )
    return {poster: encode_png(path) for poster, path in expected_paths.items()}, {
        "sample_brainmap_runtime_s": sample_runtime
    }


def render_brainmap_png(out_path: Path, flatmap: np.ndarray, masker: Any, logger: logging.Logger) -> None:
    if out_path.exists():
        return
    image = masker.inverse_transform(flatmap)
    display = plotting.plot_stat_map(
        image,
        threshold="auto",
        display_mode="ortho",
        colorbar=False,
        annotate=False,
        draw_cross=False,
        dim=-0.2,
    )
    display.savefig(str(out_path), dpi=110)
    display.close()
    logger.info("Rendered %s", out_path.name)


def encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def cluster_embeddings(specter_embeddings: np.ndarray, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    count = specter_embeddings.shape[0]
    if count < 5:
        coords = np.zeros((count, 2), dtype=np.float32)
        if count:
            coords[:, 0] = specter_embeddings[:, 0]
            if specter_embeddings.shape[1] > 1:
                coords[:, 1] = specter_embeddings[:, 1]
        labels = np.full(count, -1, dtype=np.int32)
        return coords, labels
    n_neighbors = max(2, min(15, count - 1))
    logger.info("UMAP n_neighbors=%s for %s items", n_neighbors, count)
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="cosine",
        random_state=SAMPLE_SEED,
        transform_seed=SAMPLE_SEED,
        init="random",
    )
    umap_coords = reducer.fit_transform(specter_embeddings)

    min_cluster_size = max(3, min(8, count // 8 if count >= 8 else 3))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(umap_coords)
    return umap_coords.astype(np.float32), labels.astype(np.int32)


def build_edges(coords: np.ndarray) -> list[dict[str, Any]]:
    if len(coords) < 2:
        return []
    neighbors = min(4, len(coords))
    nn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    seen: set[tuple[int, int]] = set()
    edges: list[dict[str, Any]] = []
    for src_idx, (dist_row, idx_row) in enumerate(zip(distances, indices, strict=True)):
        for distance, dst_idx in zip(dist_row[1:], idx_row[1:], strict=True):
            a, b = sorted((int(src_idx), int(dst_idx)))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append({"source": a, "target": b, "distance": float(distance)})
    return edges


def scaled_coordinates(coords: np.ndarray, width: int, height: int, padding: int = 56) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    x_span = max(float(x.max() - x.min()), 1e-6)
    y_span = max(float(y.max() - y.min()), 1e-6)
    scaled = np.zeros_like(coords, dtype=np.float32)
    scaled[:, 0] = padding + ((x - x.min()) / x_span) * (width - (padding * 2))
    scaled[:, 1] = padding + ((y - y.min()) / y_span) * (height - (padding * 2))
    return scaled


def cluster_color(label: int) -> str:
    palette = [
        "#1f4e79",
        "#b85c38",
        "#4f772d",
        "#6d597a",
        "#8d0801",
        "#355070",
        "#6b705c",
        "#9a6d38",
        "#2c7da0",
        "#7f5539",
        "#3a5a40",
        "#7b2cbf",
        "#a44a3f",
        "#006d77",
        "#7f4f24",
        "#5c677d",
        "#8a5a44",
        "#386641",
        "#5f0f40",
        "#4d6c8a",
    ]
    if label < 0:
        return "#9c8f7a"
    return palette[label % len(palette)]


def build_graph_html(
    sample_df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    brainmaps_b64: dict[str, str],
    output_path: Path,
) -> None:
    width = 1180
    height = 760
    graph_coords = scaled_coordinates(coords, width=width, height=height)
    edges = build_edges(coords)

    nodes: list[dict[str, Any]] = []
    for idx, row in enumerate(sample_df.itertuples(index=False)):
        label = int(labels[idx])
        poster = str(row.poster_number)
        nodes.append(
            {
                "id": idx,
                "poster_number": poster,
                "title": str(row.title),
                "authors": str(row.authors),
                "abstract": str(row.abstract),
                "topic_area": str(getattr(row, "topic_area", "")),
                "cluster": label,
                "cluster_name": "Uncategorized" if label < 0 else f"Cluster {label}",
                "x": round(float(graph_coords[idx, 0]), 2),
                "y": round(float(graph_coords[idx, 1]), 2),
                "color": cluster_color(label),
                "brainmap_data_url": f"data:image/png;base64,{brainmaps_b64[poster]}",
            }
        )

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    summary = {
        "sample_size": len(sample_df),
        "clusters": int(sum(cluster_counts.index >= 0)),
        "noise_points": int(cluster_counts.get(-1, 0)),
    }
    legend = [
        {
            "cluster_id": int(cluster_id),
            "label": "Uncategorized" if int(cluster_id) < 0 else f"Cluster {int(cluster_id)}",
            "count": int(count),
            "color": cluster_color(int(cluster_id)),
        }
        for cluster_id, count in cluster_counts.items()
    ]

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    legend_json = json.dumps(legend)
    summary_json = json.dumps(summary)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CNS 2026 Prototype Abstract Map (Satire)</title>
  <style>
    :root {{
      --ink: #1f1f1f;
      --muted: #5f5f5f;
      --paper: #f7f3eb;
      --panel: rgba(255, 255, 255, 0.92);
      --line: rgba(31, 31, 31, 0.12);
      --shadow: 0 18px 42px rgba(34, 34, 34, 0.09);
      --accent: #9f3a22;
      --accent-soft: rgba(159, 58, 34, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Libre Franklin", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(159, 58, 34, 0.08), transparent 28%),
        radial-gradient(circle at 88% 16%, rgba(31, 78, 121, 0.08), transparent 24%),
        linear-gradient(180deg, #fbf8f2 0%, #eef2f5 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1380px, calc(100vw - 32px));
      margin: 16px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.75fr);
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.9);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .stage {{
      padding: 22px 24px 20px;
    }}
    h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: clamp(1.8rem, 2.2vw, 2.7rem);
      line-height: 1.02;
      letter-spacing: -0.025em;
    }}
    .lede {{
      margin: 12px 0 18px;
      color: var(--muted);
      max-width: 64ch;
      line-height: 1.55;
      font-size: 1rem;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 0.8rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    .deck {{
      display: grid;
      gap: 2px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .chip {{
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.88rem;
      font-weight: 600;
      background: rgba(31, 31, 31, 0.05);
      color: var(--ink);
    }}
    .chart-wrap {{
      position: relative;
      overflow: hidden;
      padding: 0 14px 14px;
    }}
    .zoom-toolbar {{
      position: absolute;
      top: 14px;
      right: 28px;
      z-index: 4;
      display: flex;
      gap: 8px;
    }}
    .zoom-toolbar button {{
      border: 1px solid rgba(31, 31, 31, 0.12);
      background: rgba(255, 255, 255, 0.94);
      color: var(--ink);
      border-radius: 999px;
      min-width: 40px;
      min-height: 40px;
      padding: 0 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    .zoom-toolbar button:focus {{
      outline: none;
      border-color: rgba(159, 58, 34, 0.45);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 248, 246, 0.98)),
        repeating-linear-gradient(90deg, transparent 0, transparent 79px, rgba(31, 31, 31, 0.015) 80px);
      border: 1px solid rgba(31, 31, 31, 0.08);
      cursor: grab;
      user-select: none;
    }}
    svg.is-dragging {{
      cursor: grabbing;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .search-field {{
      display: grid;
      gap: 6px;
    }}
    .search-field label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      color: var(--muted);
    }}
    .controls input {{
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid rgba(31, 31, 31, 0.12);
      font: inherit;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }}
    .controls input:focus {{
      outline: none;
      border-color: rgba(159, 58, 34, 0.45);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }}
    .legend {{
      display: grid;
      gap: 10px;
      padding: 18px 18px 18px;
    }}
    .legend-scroll {{
      max-height: min(48vh, 420px);
      overflow-y: auto;
      padding-right: 6px;
      display: grid;
      gap: 10px;
    }}
    .legend-scroll::-webkit-scrollbar {{
      width: 10px;
    }}
    .legend-scroll::-webkit-scrollbar-thumb {{
      background: rgba(31, 31, 31, 0.18);
      border-radius: 999px;
      border: 2px solid rgba(255, 255, 255, 0.7);
    }}
    .legend h2, .detail h2 {{
      margin: 0 0 10px;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(31, 31, 31, 0.04);
      border: 1px solid transparent;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }}
    .legend-item:hover {{
      background: rgba(31, 31, 31, 0.07);
    }}
    .legend-item:focus {{
      outline: none;
      border-color: rgba(159, 58, 34, 0.45);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }}
    .legend-item.active {{
      background: rgba(159, 58, 34, 0.1);
      border-color: rgba(159, 58, 34, 0.28);
      transform: translateX(2px);
    }}
    .swatch {{
      width: 13px;
      height: 13px;
      border-radius: 999px;
      flex: none;
    }}
    .detail {{
      padding: 18px 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .detail-card {{
      border-radius: 18px;
      padding: 16px;
      background: rgba(31, 31, 31, 0.04);
      height: min(72vh, 760px);
      overflow-y: auto;
    }}
    .detail-meta {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .detail-title {{
      margin: 0 0 10px;
      font-size: 1.08rem;
      line-height: 1.3;
    }}
    .detail-authors {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 0.92rem;
    }}
    .detail img {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(31, 31, 31, 0.08);
      margin: 10px 0 12px;
      background: #fff;
    }}
    .detail-topic {{
      margin: 0 0 10px;
      font-size: 0.82rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }}
    .detail-abstract {{
      margin: 0;
      font-size: 0.92rem;
      line-height: 1.52;
      color: #2c2c2c;
      max-height: 240px;
      overflow: auto;
    }}
    .detail-empty {{
      display: grid;
      gap: 14px;
    }}
    .detail-empty-art {{
      display: grid;
      gap: 14px;
      margin-top: 4px;
    }}
    .skeleton-line {{
      display: block;
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(31, 31, 31, 0.06), rgba(31, 31, 31, 0.12), rgba(31, 31, 31, 0.06));
    }}
    .skeleton-line-lg {{ height: 14px; }}
    .w-92 {{ width: 92%; }}
    .w-86 {{ width: 86%; }}
    .w-74 {{ width: 74%; }}
    .w-68 {{ width: 68%; }}
    .w-58 {{ width: 58%; }}
    .detail-empty-topic {{
      width: 42%;
      height: 12px;
      border-radius: 999px;
      background: rgba(159, 58, 34, 0.16);
    }}
    .detail-empty-figure {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 16px;
      border-radius: 16px;
      min-height: 148px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.58), rgba(255, 255, 255, 0.28)),
        rgba(31, 31, 31, 0.035);
      border: 1px solid rgba(31, 31, 31, 0.06);
    }}
    .detail-empty-blob {{
      align-self: center;
      justify-self: center;
      width: clamp(64px, 26vw, 92px);
      aspect-ratio: 1 / 1;
      border-radius: 38% 44% 40% 46%;
      background:
        radial-gradient(circle at 38% 36%, rgba(196, 69, 40, 0.22), transparent 22%),
        radial-gradient(circle at 64% 58%, rgba(31, 78, 121, 0.12), transparent 24%),
        linear-gradient(180deg, rgba(255,255,255,0.85), rgba(225, 229, 235, 0.95));
      border: 1px solid rgba(31, 31, 31, 0.08);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
      opacity: 0.9;
    }}
    .detail-empty-blob:nth-child(2) {{
      border-radius: 46% 34% 44% 40%;
      width: clamp(66px, 28vw, 98px);
    }}
    .detail-empty-blob:nth-child(3) {{
      border-radius: 50%;
      width: clamp(68px, 29vw, 100px);
    }}
    .detail-empty-lines {{
      display: grid;
      gap: 10px;
    }}
    .tooltip {{
      position: absolute;
      width: min(320px, calc(100vw - 36px));
      pointer-events: none;
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 120ms ease, transform 120ms ease;
      background: rgba(26, 26, 26, 0.96);
      color: #fff;
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 22px 48px rgba(8, 8, 8, 0.28);
      z-index: 5;
    }}
    .tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    .tooltip .small {{
      margin: 0 0 6px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(255, 255, 255, 0.74);
    }}
    .tooltip h3 {{
      margin: 0 0 8px;
      font-size: 0.96rem;
      line-height: 1.35;
    }}
    .tooltip img {{
      width: 100%;
      margin-top: 8px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #fff;
    }}
    .help {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }}
    .reference-note {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.85rem;
      line-height: 1.45;
    }}
    .reference-note a {{
      color: var(--accent);
      font-weight: 700;
      text-decoration-thickness: 1px;
      text-underline-offset: 2px;
    }}
    @media (max-width: 980px) {{
      .shell {{
        width: min(100vw - 16px, 1380px);
        margin: 8px auto;
        grid-template-columns: 1fr;
      }}
      .stage {{ padding: 18px 18px 16px; }}
      .controls {{ grid-template-columns: 1fr; }}
      .chart-wrap {{ padding: 0 8px 10px; }}
      .zoom-toolbar {{
        right: 18px;
        top: 10px;
      }}
      .zoom-toolbar button {{
        min-width: 42px;
        min-height: 42px;
      }}
      .detail {{ padding: 16px 16px 18px; }}
      .detail-card {{ height: min(60vh, 680px); }}
      .legend {{ padding: 12px 16px 16px; }}
      .legend-scroll {{ max-height: 240px; }}
      .tooltip {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="card">
      <div class="stage">
        <div class="deck">
          <p class="eyebrow">Conference Abstract Atlas</p>
          <h1>CNS 2026 abstract map: If I only had a brain! (Satire)</h1>
        </div>
        <p class="lede"> Have you ever thought what your abstract would look like as a brain map? Think no longer, <a href="https://neurovlm.github.io/neurovlm/" target="_blank" rel="noreferrer">NeuroVLM</a> has got you covered!</p>
        <p class="methods"><b>Methods:</b> Each point is an abstract positioned in 2D from <a href="https://arxiv.org/abs/2004.07180" target="_blank" rel="noreferrer">SPECTER</a> text embeddings, colored by <a href="https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14" target="_blank" rel="noreferrer">HDBSCAN</a> cluster, and paired with a NeuroVLM-generated brainmap preview. Hover to scan, click to hold a card open, and use the three filters to narrow the landscape.</p>
        <div class="chips" id="summary"></div>
        <div class="controls">
          <div class="search-field">
            <label for="search-poster">Poster Number</label>
            <input id="search-poster" type="search" placeholder="e.g. A14 or F129">
          </div>
          <div class="search-field">
            <label for="search-title">Title</label>
            <input id="search-title" type="search" placeholder="Filter by keywords in title">
          </div>
          <div class="search-field">
            <label for="search-topic">Topic Area</label>
            <input id="search-topic" type="search" placeholder="Filter by topic area">
          </div>
        </div>
        <p class="help">Faint links connect nearest semantic neighbors in the UMAP layout. <strong>Uncategorized</strong> marks abstracts that HDBSCAN did not assign to a stable cluster.</p>
        <p class="reference-note">Reference: brainmap previews are generated with <a href="https://neurovlm.github.io/neurovlm/" target="_blank" rel="noreferrer">NeuroVLM</a>.</p>
      </div>
      <div class="chart-wrap">
        <div class="zoom-toolbar" aria-label="Graph zoom controls">
          <button id="zoom-in" type="button" aria-label="Zoom in">+</button>
          <button id="zoom-out" type="button" aria-label="Zoom out">-</button>
          <button id="zoom-reset" type="button" aria-label="Reset zoom">Reset</button>
        </div>
        <div class="tooltip" id="tooltip"></div>
        <svg id="graph" viewBox="0 0 {width} {height}" aria-label="UMAP cluster map"></svg>
      </div>
    </section>
    <aside class="card">
      <div class="detail">
        <h2>Selected abstract</h2>
        <div class="detail-card" id="detail-card"></div>
      </div>
      <div class="legend">
        <h2>Cluster legend</h2>
        <div class="legend-scroll" id="legend"></div>
      </div>
    </aside>
  </div>
  <script>
    const nodes = {nodes_json};
    const edges = {edges_json};
    const legend = {legend_json};
    const summary = {summary_json};
    const svg = document.getElementById("graph");
    const tooltip = document.getElementById("tooltip");
    const legendRoot = document.getElementById("legend");
    const summaryRoot = document.getElementById("summary");
    const detailCard = document.getElementById("detail-card");
    const zoomInButton = document.getElementById("zoom-in");
    const zoomOutButton = document.getElementById("zoom-out");
    const zoomResetButton = document.getElementById("zoom-reset");
    const searchPoster = document.getElementById("search-poster");
    const searchTitle = document.getElementById("search-title");
    const searchTopic = document.getElementById("search-topic");
    const baseViewBox = {{ x: 0, y: 0, w: {width}, h: {height} }};
    const minZoomWidth = baseViewBox.w * 0.18;
    let viewBox = {{ ...baseViewBox }};
    let pinnedNodeId = null;
    let activeClusterId = null;
    let dragState = null;
    let dragMoved = false;
    let lastDragAt = 0;

    function escapeHtml(value) {{
      return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function drawSummary() {{
      const items = [
        `${{summary.sample_size}} abstracts`,
        `${{summary.clusters}} clusters`,
        `${{summary.noise_points}} uncategorized`
      ];
      items.forEach((label) => {{
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = label;
        summaryRoot.appendChild(chip);
      }});
    }}

    function drawLegend() {{
      legend.forEach((item) => {{
        const row = document.createElement("div");
        row.className = "legend-item";
        row.tabIndex = 0;
        row.setAttribute("role", "button");
        row.dataset.clusterId = String(item.cluster_id);
        row.innerHTML = `
          <span class="swatch" style="background:${{item.color}}"></span>
          <strong>${{item.label}}</strong>
          <span style="margin-left:auto;color:#415a77">${{item.count}}</span>
        `;
        const toggleCluster = () => {{
          activeClusterId = activeClusterId === item.cluster_id ? null : item.cluster_id;
          pinnedNodeId = null;
          updateDetail(null);
          updateLegendSelection();
          render();
        }};
        row.addEventListener("click", toggleCluster);
        row.addEventListener("keydown", (event) => {{
          if (event.key === "Enter" || event.key === " ") {{
            event.preventDefault();
            toggleCluster();
          }}
        }});
        legendRoot.appendChild(row);
      }});
    }}

    function updateLegendSelection() {{
      Array.from(legendRoot.children).forEach((row) => {{
        const isActive = row.dataset.clusterId === String(activeClusterId);
        row.classList.toggle("active", isActive);
        row.setAttribute("aria-pressed", isActive ? "true" : "false");
      }});
    }}

    function emptyDetailMarkup() {{
      return `
        <div class="detail-empty">
          <div>
            <p class="detail-meta">Nothing selected yet</p>
            <p class="detail-abstract">Hover or click a node to inspect its poster number, title, topic area, and generated brainmap.</p>
          </div>
          <div class="detail-empty-art" aria-hidden="true">
            <span class="skeleton-line skeleton-line-lg w-92"></span>
            <span class="skeleton-line skeleton-line-lg w-74"></span>
            <span class="skeleton-line w-58"></span>
            <div class="detail-empty-topic"></div>
            <div class="detail-empty-figure">
              <span class="detail-empty-blob"></span>
              <span class="detail-empty-blob"></span>
              <span class="detail-empty-blob"></span>
            </div>
            <div class="detail-empty-lines">
              <span class="skeleton-line w-92"></span>
              <span class="skeleton-line w-86"></span>
              <span class="skeleton-line w-92"></span>
              <span class="skeleton-line w-68"></span>
            </div>
          </div>
        </div>
      `;
    }}

    function updateDetail(node) {{
      if (!node) {{
        detailCard.innerHTML = emptyDetailMarkup();
        return;
      }}
      detailCard.innerHTML = `
        <p class="detail-meta">${{escapeHtml(node.poster_number)}} • ${{escapeHtml(node.cluster_name)}}</p>
        <h3 class="detail-title">${{escapeHtml(node.title)}}</h3>
        <p class="detail-authors">${{escapeHtml(node.authors)}}</p>
        <p class="detail-topic">${{escapeHtml(node.topic_area || "No topic area listed")}}</p>
        <img src="${{node.brainmap_data_url}}" alt="Brainmap for ${{escapeHtml(node.poster_number)}}">
        <p class="detail-abstract">${{escapeHtml(node.abstract)}}</p>
      `;
    }}

    function showTooltip(node, event) {{
      tooltip.innerHTML = `
        <p class="small">${{escapeHtml(node.poster_number)}} • ${{escapeHtml(node.cluster_name)}}</p>
        <h3>${{escapeHtml(node.title)}}</h3>
        <p class="small" style="margin:0;color:rgba(255,255,255,0.65)">${{escapeHtml(node.topic_area || "No topic area listed")}}</p>
        <img src="${{node.brainmap_data_url}}" alt="Brainmap preview">
      `;
      const bounds = svg.getBoundingClientRect();
      tooltip.style.left = `${{Math.min(event.clientX - bounds.left + 18, bounds.width - 300)}}px`;
      tooltip.style.top = `${{Math.max(event.clientY - bounds.top - 14, 18)}}px`;
      tooltip.classList.add("visible");
    }}

    function hideTooltip() {{
      tooltip.classList.remove("visible");
    }}

    function clampViewBox() {{
      if (viewBox.w < minZoomWidth) {{
        viewBox.w = minZoomWidth;
        viewBox.h = baseViewBox.h * (viewBox.w / baseViewBox.w);
      }}
      if (viewBox.w > baseViewBox.w) {{
        viewBox = {{ ...baseViewBox }};
      }}
      const maxX = baseViewBox.x + baseViewBox.w - viewBox.w;
      const maxY = baseViewBox.y + baseViewBox.h - viewBox.h;
      viewBox.x = Math.min(Math.max(viewBox.x, baseViewBox.x), maxX);
      viewBox.y = Math.min(Math.max(viewBox.y, baseViewBox.y), maxY);
    }}

    function applyViewBox() {{
      clampViewBox();
      svg.setAttribute("viewBox", `${{viewBox.x}} ${{viewBox.y}} ${{viewBox.w}} ${{viewBox.h}}`);
    }}

    function eventToGraphPoint(event) {{
      const bounds = svg.getBoundingClientRect();
      const relX = (event.clientX - bounds.left) / bounds.width;
      const relY = (event.clientY - bounds.top) / bounds.height;
      return {{
        x: viewBox.x + (relX * viewBox.w),
        y: viewBox.y + (relY * viewBox.h)
      }};
    }}

    function zoomAt(pointX, pointY, factor) {{
      const nextWidth = Math.min(baseViewBox.w, Math.max(minZoomWidth, viewBox.w * factor));
      const nextHeight = baseViewBox.h * (nextWidth / baseViewBox.w);
      const relX = (pointX - viewBox.x) / viewBox.w;
      const relY = (pointY - viewBox.y) / viewBox.h;
      viewBox = {{
        x: pointX - (relX * nextWidth),
        y: pointY - (relY * nextHeight),
        w: nextWidth,
        h: nextHeight
      }};
      applyViewBox();
    }}

    function zoomByFactor(factor) {{
      const centerX = viewBox.x + (viewBox.w / 2);
      const centerY = viewBox.y + (viewBox.h / 2);
      zoomAt(centerX, centerY, factor);
    }}

    function resetZoom() {{
      viewBox = {{ ...baseViewBox }};
      applyViewBox();
    }}

    function currentFilters() {{
      return {{
        poster: searchPoster.value.trim().toLowerCase(),
        title: searchTitle.value.trim().toLowerCase(),
        topic: searchTopic.value.trim().toLowerCase(),
        cluster: activeClusterId
      }};
    }}

    function render() {{
      svg.innerHTML = "";
      const filters = currentFilters();
      const visibleNodes = nodes.filter((node) =>
        (filters.cluster === null || node.cluster === filters.cluster) &&
        (!filters.poster || node.poster_number.toLowerCase().includes(filters.poster)) &&
        (!filters.title || node.title.toLowerCase().includes(filters.title)) &&
        (!filters.topic || (node.topic_area || "").toLowerCase().includes(filters.topic))
      );
      const visibleIds = new Set(visibleNodes.map((node) => node.id));

      if (pinnedNodeId !== null && !visibleIds.has(pinnedNodeId)) {{
        pinnedNodeId = null;
        updateDetail(null);
      }}

      edges
        .filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
        .forEach((edge) => {{
          const source = nodes[edge.source];
          const target = nodes[edge.target];
          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", source.x);
          line.setAttribute("y1", source.y);
          line.setAttribute("x2", target.x);
          line.setAttribute("y2", target.y);
          line.setAttribute("stroke", "rgba(20, 33, 61, 0.12)");
          line.setAttribute("stroke-width", Math.max(0.7, 2.3 - (edge.distance * 1.6)).toFixed(2));
          svg.appendChild(line);
        }});

      visibleNodes.forEach((node) => {{
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.id === pinnedNodeId ? "10.5" : "7.5");
        circle.setAttribute("fill", node.color);
        circle.setAttribute("stroke", node.id === pinnedNodeId ? "#14213d" : "rgba(255,255,255,0.92)");
        circle.setAttribute("stroke-width", node.id === pinnedNodeId ? "3" : "2");
        circle.style.cursor = "pointer";

        circle.addEventListener("pointerdown", (event) => {{
          event.stopPropagation();
        }});

        circle.addEventListener("mouseenter", (event) => {{
          showTooltip(node, event);
          if (pinnedNodeId === null) {{
            updateDetail(node);
          }}
        }});
        circle.addEventListener("mousemove", (event) => showTooltip(node, event));
        circle.addEventListener("mouseleave", () => {{
          hideTooltip();
          if (pinnedNodeId === null) {{
            updateDetail(null);
          }}
        }});
        circle.addEventListener("click", () => {{
          if (Date.now() - lastDragAt < 220) {{
            return;
          }}
          pinnedNodeId = pinnedNodeId === node.id ? null : node.id;
          updateDetail(pinnedNodeId === null ? null : node);
          render();
        }});
        svg.appendChild(circle);
      }});
      applyViewBox();
    }}

    [searchPoster, searchTitle, searchTopic].forEach((input) => {{
      input.addEventListener("input", () => {{
        pinnedNodeId = null;
        updateDetail(null);
        render();
      }});
    }});

    svg.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const point = eventToGraphPoint(event);
      zoomAt(point.x, point.y, event.deltaY < 0 ? 0.85 : 1.18);
    }}, {{ passive: false }});

    svg.addEventListener("pointerdown", (event) => {{
      if (event.pointerType === "mouse" && event.button !== 0) {{
        return;
      }}
      dragState = {{
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
        startX: viewBox.x,
        startY: viewBox.y
      }};
      dragMoved = false;
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-dragging");
    }});

    svg.addEventListener("pointermove", (event) => {{
      if (!dragState || dragState.pointerId !== event.pointerId) {{
        return;
      }}
      const bounds = svg.getBoundingClientRect();
      const deltaX = event.clientX - dragState.clientX;
      const deltaY = event.clientY - dragState.clientY;
      if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {{
        dragMoved = true;
      }}
      viewBox.x = dragState.startX - ((deltaX / bounds.width) * viewBox.w);
      viewBox.y = dragState.startY - ((deltaY / bounds.height) * viewBox.h);
      applyViewBox();
    }});

    function finishDrag(event) {{
      if (!dragState || dragState.pointerId !== event.pointerId) {{
        return;
      }}
      if (dragMoved) {{
        lastDragAt = Date.now();
      }}
      dragState = null;
      dragMoved = false;
      svg.classList.remove("is-dragging");
    }}

    svg.addEventListener("pointerup", finishDrag);
    svg.addEventListener("pointercancel", finishDrag);

    zoomInButton.addEventListener("click", () => zoomByFactor(0.82));
    zoomOutButton.addEventListener("click", () => zoomByFactor(1.2));
    zoomResetButton.addEventListener("click", resetZoom);

    drawSummary();
    drawLegend();
    updateLegendSelection();
    updateDetail(null);
    applyViewBox();
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def human_readable_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def publish_github_pages_bundle(
    bundle_dir: Path,
    graph_html: Path,
    summary_json: Path,
    cluster_csv: Path,
    sample_csv: Path,
    structured_csv: Path,
    embeddings_npz: Path,
) -> None:
    data_dir = bundle_dir / "data"
    code_dir = bundle_dir / "code"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(graph_html, bundle_dir / "index.html")
    shutil.copy2(summary_json, data_dir / summary_json.name)
    shutil.copy2(cluster_csv, data_dir / cluster_csv.name)
    shutil.copy2(sample_csv, data_dir / sample_csv.name)
    shutil.copy2(structured_csv, data_dir / structured_csv.name)
    shutil.copy2(embeddings_npz, data_dir / embeddings_npz.name)
    for source in (
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "scripts" / "build_cns_2026_html.py",
        Path(__file__).resolve(),
    ):
        if source.exists():
            shutil.copy2(source, code_dir / source.name)

    readme = bundle_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# CNS 2026 Abstract Map",
                "",
                "This directory is the GitHub Pages bundle for the CNS 2026 abstract map.",
                "",
                "Treat `neurovlm_cns/` as the project root when rebuilding.",
                "",
                "From that root, run:",
                "",
                "```bash",
                "python scripts/build_cns_2026_html.py",
                "```",
                "",
                "The generated page in this bundle is `index.html`.",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / ".nojekyll").write_text("", encoding="utf-8")


def write_summary(
    output_path: Path,
    all_abstracts: pd.DataFrame,
    sample_df: pd.DataFrame,
    stage_times: dict[str, float],
    embedding_times: dict[str, float],
    brainmap_metrics: dict[str, float],
    sample_strategy: str,
) -> None:
    sample_size = max(1, len(sample_df))
    total_count = len(all_abstracts)
    representative_embedding_s = embedding_times.get("sample_embedding_runtime_s", stage_times["embedding_runtime_s"])
    representative_brainmap_s = brainmap_metrics.get("sample_brainmap_runtime_s", stage_times["brainmap_runtime_s"])
    per_abstract_work = (
        representative_embedding_s + representative_brainmap_s
    ) / sample_size
    estimated_full_runtime_s = stage_times["parse_runtime_s"] + (per_abstract_work * total_count) + stage_times["cluster_runtime_s"]
    payload = {
        "parsed_abstract_count": total_count,
        "sample_size": sample_size,
        "sample_strategy": sample_strategy,
        "parse_runtime_s": round(stage_times["parse_runtime_s"], 3),
        "embedding_runtime_s": round(stage_times["embedding_runtime_s"], 3),
        "brainmap_runtime_s": round(stage_times["brainmap_runtime_s"], 3),
        "cluster_runtime_s": round(stage_times["cluster_runtime_s"], 3),
        "html_runtime_s": round(stage_times["html_runtime_s"], 3),
        "representative_embedding_runtime_s": round(representative_embedding_s, 3),
        "representative_brainmap_runtime_s": round(representative_brainmap_s, 3),
        "specter_model_load_s": round(embedding_times["specter_model_load_s"], 3),
        "neurovlm_model_load_s": round(embedding_times["neurovlm_model_load_s"], 3),
        "chunk_inference_s": round(embedding_times["chunk_inference_s"], 3),
        "estimated_full_runtime_s": round(estimated_full_runtime_s, 3),
        "estimated_full_runtime_human": human_readable_seconds(estimated_full_runtime_s),
        "notes": [
            "Estimate is a warm-cache linear extrapolation from the sampled embedding and brainmap stages.",
            "First-run Hugging Face downloads will add one-time startup cost that is not included in the warm-cache estimate.",
            "The parser caches the full parsed CSV, so re-runs mainly pay for embedding, rendering, and clustering.",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def choose_sample(all_abstracts: pd.DataFrame, sample_size: int, strategy: str) -> pd.DataFrame:
    sample_size = min(sample_size, len(all_abstracts))
    if strategy == "random":
        return all_abstracts.sample(n=sample_size, random_state=SAMPLE_SEED).sort_values("poster_number").reset_index(drop=True)
    return all_abstracts.head(sample_size).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    logger.info("Arguments: %s", vars(args))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stage_times: dict[str, float] = {}

    parsed_csv = PARSED_CSV
    with timed("parse-pdf", logger) as timer:
        if parsed_csv.exists() and not args.force:
            logger.info("Loading cached parsed abstracts from %s", parsed_csv)
            all_abstracts = pd.read_csv(parsed_csv)
        else:
            all_abstracts = parse_pdf_abstracts(args.pdf, logger)
            all_abstracts.to_csv(parsed_csv, index=False)
            logger.info("Wrote parsed abstracts to %s", parsed_csv)
    stage_times["parse_runtime_s"] = timer.elapsed
    structured_csv = OUTPUT_DIR / "cns_2026_abstracts_structured.csv"
    all_abstracts.to_csv(structured_csv, index=False)
    logger.info("Wrote full structured CSV to %s", structured_csv)

    sample_df = choose_sample(all_abstracts, args.sample_size, args.sample_strategy)
    sample_tag = f"sample_{len(sample_df):03d}_{args.sample_strategy}"
    sample_csv = OUTPUT_DIR / f"cns_2026_{sample_tag}_abstracts.csv"
    sample_df.to_csv(sample_csv, index=False)
    logger.info("Wrote sample CSV to %s", sample_csv)

    embedding_cache = CACHE_DIR / f"{sample_tag}_embeddings.npz"
    with timed("embed-text", logger) as timer:
        specter_embeddings, neurovlm_embeddings, embedding_times = build_embeddings(
            sample_df=sample_df,
            batch_size=args.batch_size,
            chunk_words=args.chunk_words,
            cache_path=embedding_cache,
            force=args.force,
            logger=logger,
        )
    stage_times["embedding_runtime_s"] = timer.elapsed

    brainmap_dir = CACHE_DIR / f"{sample_tag}_brainmaps"
    with timed("render-brainmaps", logger) as timer:
        brainmaps_b64, brainmap_metrics = decode_brainmaps(
            neurovlm_embeddings=neurovlm_embeddings,
            sample_df=sample_df,
            brainmap_dir=brainmap_dir,
            force=args.force,
            workers=args.brainmap_workers,
            logger=logger,
        )
    stage_times["brainmap_runtime_s"] = timer.elapsed

    with timed("cluster", logger) as timer:
        umap_coords, labels = cluster_embeddings(specter_embeddings, logger)
        cluster_csv = OUTPUT_DIR / f"cns_2026_{sample_tag}_clusters.csv"
        clustered = sample_df.copy()
        clustered["umap_x"] = umap_coords[:, 0]
        clustered["umap_y"] = umap_coords[:, 1]
        clustered["cluster"] = labels
        clustered.to_csv(cluster_csv, index=False)
        logger.info("Wrote clustering CSV to %s", cluster_csv)
    stage_times["cluster_runtime_s"] = timer.elapsed

    embedding_npz = OUTPUT_DIR / f"cns_2026_{sample_tag}_embeddings.npz"
    save_npz(
        embedding_npz,
        poster_numbers=sample_df["poster_number"].astype(str).to_numpy(dtype="U32"),
        specter_embeddings=specter_embeddings.astype(np.float32),
        neurovlm_embeddings=neurovlm_embeddings.astype(np.float32),
        umap_coordinates=umap_coords.astype(np.float32),
        cluster_labels=labels.astype(np.int32),
    )
    logger.info("Wrote output embeddings package to %s", embedding_npz)

    graph_html = OUTPUT_DIR / f"cns_2026_{sample_tag}_map.html"
    with timed("build-html", logger) as timer:
        build_graph_html(
            sample_df=sample_df,
            coords=umap_coords,
            labels=labels,
            brainmaps_b64=brainmaps_b64,
            output_path=graph_html,
        )
        logger.info("Wrote interactive graph to %s", graph_html)
    stage_times["html_runtime_s"] = timer.elapsed

    summary_json = OUTPUT_DIR / f"cns_2026_{sample_tag}_summary.json"
    write_summary(
        output_path=summary_json,
        all_abstracts=all_abstracts,
        sample_df=sample_df,
        stage_times=stage_times,
        embedding_times=embedding_times,
        brainmap_metrics=brainmap_metrics,
        sample_strategy=args.sample_strategy,
    )
    logger.info("Wrote summary to %s", summary_json)

    publish_dir = PROJECT_ROOT / "docs" / "cns_2026_map"
    publish_github_pages_bundle(
        bundle_dir=publish_dir,
        graph_html=graph_html,
        summary_json=summary_json,
        cluster_csv=cluster_csv,
        sample_csv=sample_csv,
        structured_csv=structured_csv,
        embeddings_npz=embedding_npz,
    )
    logger.info("Published GitHub Pages bundle to %s", publish_dir)
    logger.info("Prototype complete: parsed=%s sample=%s", len(all_abstracts), len(sample_df))


if __name__ == "__main__":
    main()
