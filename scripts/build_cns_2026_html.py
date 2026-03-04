#!/usr/bin/env python3
"""Rebuild the CNS 2026 HTML map from cached clustering data and brainmap PNGs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache" / "cns_2026_prototype"
DOCS_DIR = PROJECT_ROOT / "docs" / "cns_2026_map"
TRANSPARENT_GIF_DATA_URL = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
CLUSTER_NAME_RE = re.compile(r"cns_2026_(sample_\d+_[a-z]+)_clusters\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-csv",
        type=Path,
        default=OUTPUT_DIR / "cns_2026_sample_769_head_clusters.csv",
        help="Cluster CSV produced by the main pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to the matching *_map.html beside the cluster CSV.",
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Only write the output HTML; do not refresh docs/cns_2026_map.",
    )
    return parser.parse_args()


def sample_tag_from_cluster_csv(cluster_csv: Path) -> str:
    match = CLUSTER_NAME_RE.fullmatch(cluster_csv.name)
    if not match:
        raise ValueError(
            "Cluster CSV name must match cns_2026_<sample_tag>_clusters.csv "
            f"so the brainmap cache can be inferred. Got: {cluster_csv.name}"
        )
    return match.group(1)


def default_output_path(cluster_csv: Path) -> Path:
    sample_tag = sample_tag_from_cluster_csv(cluster_csv)
    return cluster_csv.with_name(f"cns_2026_{sample_tag}_map.html")


def infer_brainmap_dir(sample_tag: str) -> Path:
    return CACHE_DIR / f"{sample_tag}_brainmaps"


def load_cluster_rows(cluster_csv: Path) -> list[dict[str, str]]:
    with cluster_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {cluster_csv}")
    required = {"poster_number", "title", "authors", "abstract", "umap_x", "umap_y", "cluster"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns in {cluster_csv}: {sorted(missing)}")
    return rows


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


def scaled_coordinates(coords: list[tuple[float, float]], width: int, height: int, padding: int = 56) -> list[tuple[float, float]]:
    if not coords:
        return []
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    x_min = min(xs)
    y_min = min(ys)
    x_span = max(max(xs) - x_min, 1e-6)
    y_span = max(max(ys) - y_min, 1e-6)
    scale_x = width - (padding * 2)
    scale_y = height - (padding * 2)
    return [
        (
            round(padding + (((x - x_min) / x_span) * scale_x), 2),
            round(padding + (((y - y_min) / y_span) * scale_y), 2),
        )
        for x, y in coords
    ]


def build_edges(coords: list[tuple[float, float]]) -> list[dict[str, float | int]]:
    if len(coords) < 2:
        return []
    neighbors = min(4, len(coords))
    edges: list[dict[str, float | int]] = []
    seen: set[tuple[int, int]] = set()
    for src_idx, (src_x, src_y) in enumerate(coords):
        distances: list[tuple[float, int]] = []
        for dst_idx, (dst_x, dst_y) in enumerate(coords):
            distance = math.hypot(src_x - dst_x, src_y - dst_y)
            distances.append((distance, dst_idx))
        distances.sort(key=lambda item: item[0])
        for distance, dst_idx in distances[1:neighbors]:
            edge = tuple(sorted((src_idx, dst_idx)))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append({"source": edge[0], "target": edge[1], "distance": round(distance, 6)})
    return edges


def load_brainmap_urls(rows: list[dict[str, str]], brainmap_dir: Path) -> dict[str, str]:
    brainmaps: dict[str, str] = {}
    for row in rows:
        poster = row["poster_number"]
        image_path = brainmap_dir / f"{poster}.png"
        if image_path.exists():
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            brainmaps[poster] = f"data:image/png;base64,{encoded}"
        else:
            brainmaps[poster] = TRANSPARENT_GIF_DATA_URL
    return brainmaps


def build_graph_html(
    rows: list[dict[str, str]],
    coords: list[tuple[float, float]],
    labels: list[int],
    brainmaps: dict[str, str],
    output_path: Path,
) -> None:
    width = 1180
    height = 760
    graph_coords = scaled_coordinates(coords, width=width, height=height)
    edges = build_edges(coords)

    nodes: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        label = labels[idx]
        poster = row["poster_number"]
        topic_area = row.get("topic_area", "")
        nodes.append(
            {
                "id": idx,
                "poster_number": poster,
                "title": row["title"],
                "authors": row["authors"],
                "abstract": row["abstract"],
                "topic_area": topic_area,
                "cluster": label,
                "cluster_name": "Uncategorized" if label < 0 else f"Cluster {label}",
                "x": graph_coords[idx][0],
                "y": graph_coords[idx][1],
                "color": cluster_color(label),
                "brainmap_data_url": brainmaps[poster],
            }
        )

    cluster_counts = Counter(labels)
    ordered_cluster_ids = sorted(cluster_counts)
    summary = {
        "sample_size": len(rows),
        "clusters": sum(1 for cluster_id in ordered_cluster_ids if cluster_id >= 0),
        "noise_points": cluster_counts.get(-1, 0),
    }
    legend = [
        {
            "cluster_id": cluster_id,
            "label": "Uncategorized" if cluster_id < 0 else f"Cluster {cluster_id}",
            "count": cluster_counts[cluster_id],
            "color": cluster_color(cluster_id),
        }
        for cluster_id in ordered_cluster_ids
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
      svg.releasePointerCapture(event.pointerId);
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def publish_bundle(output_html: Path, cluster_csv: Path, sample_tag: str) -> None:
    data_dir = DOCS_DIR / "data"
    code_dir = DOCS_DIR / "code"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(output_html, DOCS_DIR / "index.html")
    copy_if_exists(cluster_csv, data_dir / cluster_csv.name)
    copy_if_exists(OUTPUT_DIR / "cns_2026_abstracts_structured.csv", data_dir / "cns_2026_abstracts_structured.csv")
    copy_if_exists(OUTPUT_DIR / f"cns_2026_{sample_tag}_abstracts.csv", data_dir / f"cns_2026_{sample_tag}_abstracts.csv")
    copy_if_exists(OUTPUT_DIR / f"cns_2026_{sample_tag}_summary.json", data_dir / f"cns_2026_{sample_tag}_summary.json")
    copy_if_exists(OUTPUT_DIR / f"cns_2026_{sample_tag}_embeddings.npz", data_dir / f"cns_2026_{sample_tag}_embeddings.npz")

    copy_if_exists(PROJECT_ROOT / "pyproject.toml", code_dir / "pyproject.toml")
    copy_if_exists(PROJECT_ROOT / "scripts" / "build_cns_2026_html.py", code_dir / "build_cns_2026_html.py")
    copy_if_exists(PROJECT_ROOT / "scripts" / "cns_2026_prototype_pipeline.py", code_dir / "cns_2026_prototype_pipeline.py")

    readme = DOCS_DIR / "README.md"
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
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cluster_csv = args.cluster_csv.resolve()
    if not cluster_csv.exists():
        raise FileNotFoundError(f"Cluster CSV not found: {cluster_csv}")

    sample_tag = sample_tag_from_cluster_csv(cluster_csv)
    output_path = args.output.resolve() if args.output else default_output_path(cluster_csv)
    brainmap_dir = infer_brainmap_dir(sample_tag)
    if not brainmap_dir.exists():
        raise FileNotFoundError(f"Brainmap cache directory not found: {brainmap_dir}")

    rows = load_cluster_rows(cluster_csv)
    coords = [(float(row["umap_x"]), float(row["umap_y"])) for row in rows]
    labels = [int(float(row["cluster"])) for row in rows]
    brainmaps = load_brainmap_urls(rows, brainmap_dir)

    build_graph_html(rows, coords, labels, brainmaps, output_path)
    if not args.skip_publish:
        publish_bundle(output_path, cluster_csv, sample_tag)

    print(f"Wrote HTML: {output_path}")
    if not args.skip_publish:
        print(f"Updated bundle: {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
