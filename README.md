# NeuroVLM CNS

Treat this directory as the project root for the CNS 2026 abstract map.

## Layout

- `CNS_2026_Abstracts.pdf`: source abstract book.
- `scripts/cns_2026_prototype_pipeline.py`: full parse/embed/brainmap/cluster pipeline.
- `scripts/build_cns_2026_html.py`: lightweight HTML-only rebuild from cached clustering CSV + brainmap PNGs.
- `output/`: generated CSV, NPZ, summaries, and the HTML map.
- `output/cache/cns_2026_prototype/`: cached parsed abstracts, embeddings, and brainmap PNGs.
- `docs/cns_2026_map/`: GitHub Pages-ready bundle. `index.html` is the page to publish.

## Install NeuroVLM

Install `neurovlm` directly from GitHub into the shared repo `.venv`:

```bash
uv pip install git+https://github.com/neurovlm/neurovlm.git
```

The full pipeline prefers the installed package first and only falls back to a local source checkout if the package is not available.

## Rebuild The HTML

The minimal rebuild path uses only the Python standard library and the cached artifacts already in `output/`.

From this directory:

```bash
python scripts/build_cns_2026_html.py
```

That command rebuilds:

- `output/cns_2026_sample_769_head_map.html`
- `docs/cns_2026_map/index.html`

To rebuild a different sample or write to a different file:

```bash
python scripts/build_cns_2026_html.py \
  --cluster-csv output/cns_2026_sample_100_head_clusters.csv \
  --output output/cns_2026_sample_100_head_map.html
```

The script infers the matching brainmap cache directory from the cluster CSV name. For example, `output/cns_2026_sample_769_head_clusters.csv` uses `output/cache/cns_2026_prototype/sample_769_head_brainmaps/`.

## Run The Full Pipeline

The full pipeline still lives in `scripts/cns_2026_prototype_pipeline.py` and writes all outputs relative to this directory.

```bash
python scripts/cns_2026_prototype_pipeline.py --sample-size 769 --brainmap-workers 2
```

That heavier path depends on the shared sibling `../neurovlm/` checkout unless you vendor a `neurovlm/` folder into this directory.
