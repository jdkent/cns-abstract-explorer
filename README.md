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

Install `neurovlm` directly from GitHub into this repo's `.venv`:

```bash
uv pip install --python .venv/bin/python git+https://github.com/neurovlm/neurovlm.git
```

The full pipeline imports only the installed package. There is no fallback to a local source checkout.

## Rebuild The HTML

The minimal rebuild path uses only the Python standard library and the cached artifacts already in `output/`.

From this directory:

```bash
.venv/bin/python scripts/build_cns_2026_html.py
```

That command rebuilds:

- `output/cns_2026_sample_769_head_map.html`
- `docs/cns_2026_map/index.html`

If you rebuild from a NeuroVLM-clustered CSV such as `output/cns_2026_sample_769_head_neurovlm_clusters.csv`, the output HTML keeps the same `_neurovlm` suffix.

To rebuild a different sample or write to a different file:

```bash
.venv/bin/python scripts/build_cns_2026_html.py \
  --cluster-csv output/cns_2026_sample_100_head_clusters.csv \
  --output output/cns_2026_sample_100_head_map.html
```

The script infers the matching brainmap cache directory from the cluster CSV name. For example, `output/cns_2026_sample_769_head_clusters.csv` uses `output/cache/cns_2026_prototype/sample_769_head_brainmaps/`.

## Run The Full Pipeline

The full pipeline still lives in `scripts/cns_2026_prototype_pipeline.py` and writes all outputs relative to this directory.

```bash
.venv/bin/python scripts/cns_2026_prototype_pipeline.py --sample-size 769 --brainmap-workers 2
```

That heavier path depends on `neurovlm` being installed in this repo's active environment.

To cluster in the NeuroVLM projected text space instead of the default SPECTER space:

```bash
.venv/bin/python scripts/cns_2026_prototype_pipeline.py \
  --sample-size 769 \
  --brainmap-workers 2 \
  --cluster-embedding-space neurovlm
```

That writes clustering-dependent artifacts with a `_neurovlm` suffix, for example:

- `output/cns_2026_sample_769_head_neurovlm_clusters.csv`
- `output/cns_2026_sample_769_head_neurovlm_map.html`
- `output/cns_2026_sample_769_head_neurovlm_summary.json`

The pipeline also exposes clustering overrides:

- `--umap-n-neighbors`
- `--umap-min-dist`
- `--hdbscan-min-cluster-size`
- `--hdbscan-min-samples`

By default, the NeuroVLM clustering profile uses a more permissive UMAP/HDBSCAN setup than the SPECTER profile so fewer abstracts are left uncategorized.
The current NeuroVLM defaults are tuned toward a coarser map with fewer clusters and lower noise: `--umap-n-neighbors 60`, `--umap-min-dist 0.02`, `--hdbscan-min-cluster-size 13`, `--hdbscan-min-samples 1`.

If `.env` contains `OPENAI_API_KEY`, the full pipeline also sends all non-noise cluster title sets to OpenAI in one global naming pass so the model can label clusters relative to each other. If that response still produces duplicate labels, the pipeline submits only the conflicting title sets again and asks for distinct names while preserving the cluster-id mapping. Those names are written into the `cluster_name` column in the clustering CSV and reused by the HTML-only rebuild, so cached rebuilds do not make extra API calls. You can optionally set `OPENAI_CLUSTER_LABEL_MODEL` in `.env`; otherwise the pipeline uses `gpt-4o-mini`.
