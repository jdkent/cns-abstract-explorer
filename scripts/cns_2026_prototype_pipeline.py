#!/usr/bin/env python3
"""Prototype CNS 2026 abstract pipeline for parsing, embedding, and visualization."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import pandas as pd
import torch
import matplotlib
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

matplotlib.use("Agg")

import hdbscan
from nilearn import plotting
from build_cns_2026_html import build_graph_html as html_build_graph, publish_bundle as publish_html_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "CNS_2026_Abstracts.pdf"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache" / "cns_2026_prototype"
SCRIPT_LOG = CACHE_DIR / "pipeline.log"
PARSED_CSV = CACHE_DIR / "parsed_abstracts.csv"
SPECTER_MODEL_ID = "allenai/specter2_aug2023refresh_base"
SAMPLE_SEED = 42
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CLUSTER_LABEL_MODEL = "gpt-4o-mini"

try:
    from neurovlm.data import load_masker  # noqa: E402
    from neurovlm.models import load_model as load_neurovlm_model  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import `neurovlm`. Install the GitHub package into this project's environment, "
        "for example with `uv pip install --python .venv/bin/python "
        "git+https://github.com/neurovlm/neurovlm.git`."
    ) from exc


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
        help="Batch size for text embedding inference. Lower this if RAM is tight.",
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
    parser.add_argument(
        "--cluster-embedding-space",
        choices=("specter", "neurovlm"),
        default="specter",
        help=(
            "Embedding space used for UMAP + clustering. "
            "`neurovlm` uses the 384-d NeuroVLM projected text space (`proj_head_text_mse`)."
        ),
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=None,
        help="Override UMAP n_neighbors for clustering. Defaults are chosen per embedding space.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=None,
        help="Override UMAP min_dist for clustering. Defaults are chosen per embedding space.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=None,
        help="Override HDBSCAN min_cluster_size. Defaults are chosen per embedding space.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="Override HDBSCAN min_samples. Defaults are chosen per embedding space.",
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


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def cluster_name_fallback(cluster_id: int) -> str:
    if cluster_id < 0:
        return "Uncategorized"
    return f"Cluster {cluster_id}"


def normalize_cluster_name(raw_text: str, cluster_id: int) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return cluster_name_fallback(cluster_id)
    candidate = lines[0]
    candidate = re.sub(r"^(?:[-*]|\d+[.):-])\s*", "", candidate)
    candidate = re.sub(r"^cluster\s+\d+\s*[:.-]\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(" \t\r\n\"'`.,;:-")
    candidate = re.sub(r"\s+", " ", candidate)
    if not candidate:
        return cluster_name_fallback(cluster_id)
    words = candidate.split()
    if len(words) > 3:
        candidate = " ".join(words[:3])
    return candidate


def extract_openai_message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def openai_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, Any] | None:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    request = urllib_request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def request_global_cluster_names(
    titles_by_cluster: dict[int, list[str]],
    api_key: str,
    model: str,
    logger: logging.Logger,
) -> dict[int, str]:
    prompt_sections = []
    for cluster_id in sorted(titles_by_cluster):
        titles = titles_by_cluster[cluster_id]
        title_lines = "\n".join(f"- {title}" for title in titles)
        prompt_sections.append(f"Cluster {cluster_id} titles:\n{title_lines}")
    prompt_text = "\n\n".join(prompt_sections)
    try:
        body = openai_chat_completion(
            api_key=api_key,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional neuroscientist who is fantastic at seeing patterns "
                        "in neuroscientific text and differentiating closely related topics. "
                        "You prefer broader umbrella themes over narrow task names, methods, or highly specific subtopics."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Below are the title sets for every non-noise cluster in a neuroscience poster map.\n"
                        "Read all clusters together and generate broad but still distinctive, descriptive, and "
                        "representative labels for every cluster.\n"
                        "Return exactly one 1-3 word label per cluster.\n"
                        "Prefer umbrella categories that summarize the dominant theme of each cluster rather than "
                        "narrow methods, paradigms, or single-study-specific details.\n"
                        "Avoid overly specific labels unless a narrow theme clearly dominates the whole cluster.\n"
                        "Make the labels globally distinct from each other, not just locally descriptive.\n"
                        "Return JSON only as an object mapping cluster ids to labels, for example "
                        '{"0":"Working Memory","1":"Auditory Attention"}.\n'
                        "Do not include explanation text.\n\n"
                        f"{prompt_text}"
                    ),
                },
            ],
            max_tokens=max(120, len(titles_by_cluster) * 12),
        )
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("OpenAI global cluster naming failed: %s", detail)
        return {}
    except urllib_error.URLError as exc:
        logger.warning("OpenAI global cluster naming failed: %s", exc)
        return {}

    if body is None:
        return {}
    parsed = parse_json_object(extract_openai_message_content(body))
    if not parsed:
        logger.warning("OpenAI global cluster naming returned non-JSON.")
        return {}

    cluster_names: dict[int, str] = {}
    for cluster_id in sorted(titles_by_cluster):
        raw_name = parsed.get(str(cluster_id), parsed.get(cluster_id))
        if raw_name is None:
            continue
        cluster_names[cluster_id] = normalize_cluster_name(str(raw_name), cluster_id)
        logger.info("Named cluster %s as %s", cluster_id, cluster_names[cluster_id])
    return cluster_names


def resolve_duplicate_cluster_names(
    cluster_names: dict[int, str],
    titles_by_cluster: dict[int, list[str]],
    api_key: str,
    model: str,
    logger: logging.Logger,
) -> dict[int, str]:
    duplicate_groups: dict[str, list[int]] = {}
    for cluster_id, name in cluster_names.items():
        duplicate_groups.setdefault(name, []).append(cluster_id)

    for duplicate_name, cluster_ids in sorted(duplicate_groups.items()):
        if len(cluster_ids) < 2:
            continue
        prompt_sections = []
        for cluster_id in sorted(cluster_ids):
            titles = titles_by_cluster.get(cluster_id, [])
            title_lines = "\n".join(f"- {title}" for title in titles)
            prompt_sections.append(f"Cluster {cluster_id} titles:\n{title_lines}")
        prompt_text = "\n\n".join(prompt_sections)
        try:
            body = openai_chat_completion(
                api_key=api_key,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional neuroscientist who is fantastic at seeing patterns "
                            "in neuroscientific text and differentiating closely related topics. "
                            "You prefer broader umbrella themes over narrow task names, methods, or highly specific subtopics."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "These clusters were previously given the same label, but they need distinct names.\n"
                            "For each cluster below, generate a different 1-3 word label that best matches that "
                            "cluster's titles while clearly distinguishing it from the others in this request.\n"
                            "Prefer broader umbrella labels rather than narrow methods, paradigms, or one-off details.\n"
                            "Keep the labels distinct, but do not overfit to minor differences.\n"
                            "Return JSON only as an object mapping cluster ids to labels, for example "
                            '{"3":"Working Memory","11":"Task Switching"}.\n'
                            "Do not include explanation text.\n\n"
                            f"{prompt_text}"
                        ),
                    },
                ],
                max_tokens=80,
            )
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.warning(
                "OpenAI duplicate-name resolution failed for %s (%s): %s",
                cluster_ids,
                duplicate_name,
                detail,
            )
            continue
        except urllib_error.URLError as exc:
            logger.warning(
                "OpenAI duplicate-name resolution failed for %s (%s): %s",
                cluster_ids,
                duplicate_name,
                exc,
            )
            continue

        if body is None:
            continue
        parsed = parse_json_object(extract_openai_message_content(body))
        if not parsed:
            logger.warning(
                "OpenAI duplicate-name resolution returned non-JSON for %s (%s).",
                cluster_ids,
                duplicate_name,
            )
            continue

        seen_names: set[str] = set()
        updated_any = False
        for cluster_id in sorted(cluster_ids):
            raw_name = parsed.get(str(cluster_id), parsed.get(cluster_id))
            if raw_name is None:
                continue
            normalized = normalize_cluster_name(str(raw_name), cluster_id)
            if normalized in seen_names:
                normalized = normalize_cluster_name(f"{normalized} {cluster_id}", cluster_id)
            seen_names.add(normalized)
            cluster_names[cluster_id] = normalized
            updated_any = True

        if updated_any:
            logger.info(
                "Resolved duplicate cluster name %s for clusters %s -> %s",
                duplicate_name,
                cluster_ids,
                {cluster_id: cluster_names[cluster_id] for cluster_id in sorted(cluster_ids)},
            )
    return cluster_names


def generate_cluster_names(sample_df: pd.DataFrame, labels: np.ndarray, logger: logging.Logger) -> dict[int, str]:
    cluster_ids = sorted({int(label) for label in labels.tolist() if int(label) >= 0})
    if not cluster_ids:
        return {}

    env_values = load_dotenv(DOTENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY") or env_values.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("No OPENAI_API_KEY found in environment or %s; using fallback cluster names.", DOTENV_PATH)
        return {}
    model = os.environ.get("OPENAI_CLUSTER_LABEL_MODEL") or env_values.get(
        "OPENAI_CLUSTER_LABEL_MODEL",
        DEFAULT_CLUSTER_LABEL_MODEL,
    )

    titles_by_cluster = {
        cluster_id: sample_df.loc[labels == cluster_id, "title"].astype(str).tolist()
        for cluster_id in cluster_ids
    }
    cluster_names = request_global_cluster_names(
        titles_by_cluster=titles_by_cluster,
        api_key=api_key,
        model=model,
        logger=logger,
    )
    for cluster_id in cluster_ids:
        cluster_names.setdefault(cluster_id, cluster_name_fallback(cluster_id))
    return resolve_duplicate_cluster_names(
        cluster_names=cluster_names,
        titles_by_cluster=titles_by_cluster,
        api_key=api_key,
        model=model,
        logger=logger,
    )


def resolve_cluster_params(
    count: int,
    embedding_space: str,
    args: argparse.Namespace,
) -> dict[str, int | float | None]:
    default_min_cluster_size = max(3, min(8, count // 8 if count >= 8 else 3))
    if embedding_space == "neurovlm":
        neurovlm_n_neighbors = min(60, max(10, count // 2), max(2, count - 1))
        neurovlm_min_cluster_size = min(13, max(4, count // 8), max(4, count - 1))
        defaults: dict[str, int | float | None] = {
            "umap_n_neighbors": neurovlm_n_neighbors,
            "umap_min_dist": 0.02,
            "hdbscan_min_cluster_size": neurovlm_min_cluster_size,
            "hdbscan_min_samples": 1,
        }
    else:
        defaults = {
            "umap_n_neighbors": max(2, min(15, count - 1)),
            "umap_min_dist": 0.15,
            "hdbscan_min_cluster_size": default_min_cluster_size,
            "hdbscan_min_samples": None,
        }

    if args.umap_n_neighbors is not None:
        defaults["umap_n_neighbors"] = max(2, min(args.umap_n_neighbors, max(2, count - 1)))
    if args.umap_min_dist is not None:
        defaults["umap_min_dist"] = args.umap_min_dist
    if args.hdbscan_min_cluster_size is not None:
        defaults["hdbscan_min_cluster_size"] = args.hdbscan_min_cluster_size
    if args.hdbscan_min_samples is not None:
        defaults["hdbscan_min_samples"] = args.hdbscan_min_samples
    return defaults


def cluster_embeddings(
    text_embeddings: np.ndarray,
    embedding_space: str,
    logger: logging.Logger,
    cluster_params: dict[str, int | float | None],
) -> tuple[np.ndarray, np.ndarray]:
    count = text_embeddings.shape[0]
    if count < 5:
        coords = np.zeros((count, 2), dtype=np.float32)
        if count:
            coords[:, 0] = text_embeddings[:, 0]
            if text_embeddings.shape[1] > 1:
                coords[:, 1] = text_embeddings[:, 1]
        labels = np.full(count, -1, dtype=np.int32)
        return coords, labels
    n_neighbors = int(cluster_params["umap_n_neighbors"])
    min_dist = float(cluster_params["umap_min_dist"])
    min_cluster_size = int(cluster_params["hdbscan_min_cluster_size"])
    min_samples = cluster_params["hdbscan_min_samples"]
    logger.info(
        "Clustering params: space=%s n_neighbors=%s min_dist=%.3f min_cluster_size=%s min_samples=%s",
        embedding_space,
        n_neighbors,
        min_dist,
        min_cluster_size,
        min_samples,
    )
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=SAMPLE_SEED,
        transform_seed=SAMPLE_SEED,
        init="random",
    )
    umap_coords = reducer.fit_transform(text_embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(umap_coords)
    return umap_coords.astype(np.float32), labels.astype(np.int32)


def build_graph_html(
    sample_df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    brainmaps_b64: dict[str, str],
    output_path: Path,
    embedding_space: str,
    cluster_names: dict[int, str] | None = None,
) -> None:
    rows: list[dict[str, str]] = []
    for row in sample_df.itertuples(index=False):
        topic_area = getattr(row, "topic_area", "")
        if pd.isna(topic_area):
            topic_area = ""
        rows.append(
            {
                "poster_number": str(row.poster_number),
                "title": str(row.title),
                "authors": str(row.authors),
                "abstract": str(row.abstract),
                "topic_area": str(topic_area),
            }
        )
    brainmap_urls = {
        poster_number: f"data:image/png;base64,{encoded}"
        for poster_number, encoded in brainmaps_b64.items()
    }
    html_build_graph(
        rows=rows,
        coords=[(float(x), float(y)) for x, y in coords.tolist()],
        labels=[int(label) for label in labels.tolist()],
        brainmaps=brainmap_urls,
        output_path=output_path,
        embedding_space=embedding_space,
        cluster_names=cluster_names,
    )


def human_readable_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def write_summary(
    output_path: Path,
    all_abstracts: pd.DataFrame,
    sample_df: pd.DataFrame,
    stage_times: dict[str, float],
    embedding_times: dict[str, float],
    brainmap_metrics: dict[str, float],
    sample_strategy: str,
    cluster_embedding_space: str,
    cluster_names: dict[int, str],
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
        "cluster_embedding_space": cluster_embedding_space,
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
        "cluster_names": {str(cluster_id): name for cluster_id, name in sorted(cluster_names.items())},
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


def artifact_tag(sample_tag: str, cluster_embedding_space: str) -> str:
    if cluster_embedding_space == "specter":
        return sample_tag
    return f"{sample_tag}_{cluster_embedding_space}"


def clustering_input_embeddings(
    specter_embeddings: np.ndarray,
    neurovlm_embeddings: np.ndarray,
    cluster_embedding_space: str,
) -> np.ndarray:
    if cluster_embedding_space == "specter":
        return specter_embeddings
    return neurovlm_embeddings


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
    run_tag = artifact_tag(sample_tag, args.cluster_embedding_space)
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
        cluster_input = clustering_input_embeddings(
            specter_embeddings=specter_embeddings,
            neurovlm_embeddings=neurovlm_embeddings,
            cluster_embedding_space=args.cluster_embedding_space,
        )
        cluster_params = resolve_cluster_params(
            count=len(sample_df),
            embedding_space=args.cluster_embedding_space,
            args=args,
        )
        umap_coords, labels = cluster_embeddings(
            text_embeddings=cluster_input,
            embedding_space=args.cluster_embedding_space,
            logger=logger,
            cluster_params=cluster_params,
        )
        cluster_csv = OUTPUT_DIR / f"cns_2026_{run_tag}_clusters.csv"
        clustered = sample_df.copy()
        clustered["umap_x"] = umap_coords[:, 0]
        clustered["umap_y"] = umap_coords[:, 1]
        clustered["cluster"] = labels
        cluster_names = generate_cluster_names(sample_df=sample_df, labels=labels, logger=logger)
        clustered["cluster_name"] = [cluster_names.get(int(label), cluster_name_fallback(int(label))) for label in labels.tolist()]
        clustered.to_csv(cluster_csv, index=False)
        logger.info("Wrote clustering CSV to %s", cluster_csv)
    stage_times["cluster_runtime_s"] = timer.elapsed

    embedding_npz = OUTPUT_DIR / f"cns_2026_{run_tag}_embeddings.npz"
    save_npz(
        embedding_npz,
        poster_numbers=sample_df["poster_number"].astype(str).to_numpy(dtype="U32"),
        cluster_embedding_space=np.array([args.cluster_embedding_space], dtype="U16"),
        specter_embeddings=specter_embeddings.astype(np.float32),
        neurovlm_embeddings=neurovlm_embeddings.astype(np.float32),
        umap_coordinates=umap_coords.astype(np.float32),
        cluster_labels=labels.astype(np.int32),
    )
    logger.info("Wrote output embeddings package to %s", embedding_npz)

    graph_html = OUTPUT_DIR / f"cns_2026_{run_tag}_map.html"
    with timed("build-html", logger) as timer:
        build_graph_html(
            sample_df=sample_df,
            coords=umap_coords,
            labels=labels,
            brainmaps_b64=brainmaps_b64,
            output_path=graph_html,
            embedding_space=args.cluster_embedding_space,
            cluster_names=cluster_names,
        )
        logger.info("Wrote interactive graph to %s", graph_html)
    stage_times["html_runtime_s"] = timer.elapsed

    summary_json = OUTPUT_DIR / f"cns_2026_{run_tag}_summary.json"
    write_summary(
        output_path=summary_json,
        all_abstracts=all_abstracts,
        sample_df=sample_df,
        stage_times=stage_times,
        embedding_times=embedding_times,
        brainmap_metrics=brainmap_metrics,
        sample_strategy=args.sample_strategy,
        cluster_embedding_space=args.cluster_embedding_space,
        cluster_names=cluster_names,
    )
    logger.info("Wrote summary to %s", summary_json)

    publish_dir = PROJECT_ROOT / "docs" / "cns_2026_map"
    publish_html_bundle(graph_html, cluster_csv, sample_tag)
    logger.info("Published GitHub Pages bundle to %s", publish_dir)
    logger.info("Prototype complete: parsed=%s sample=%s", len(all_abstracts), len(sample_df))


if __name__ == "__main__":
    main()
