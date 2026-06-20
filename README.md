# Batchlense

**Detect batch anomalies before they become recalls.**

Batchlense is a small quality-analytics app for manufacturing and process teams: upload batch-level CSVs, score every row with **Local Outlier Factor (LOF)** against a reference baseline, and explore results with control-style charts, **Pp/Ppk**, 2D projections (PCA / t-SNE / UMAP), and PDF export.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://batchlense.streamlit.app)

> Replace the demo URL above after you deploy your own fork.

![Dashboard demo](docs/demo.png)

*Add `docs/demo.png` after your first deploy so this image renders.*

---

## Why it exists

Production data often arrives as **one row per batch** with tens of numeric parameters. Spotting a bad batch in a spreadsheet is slow and inconsistent. Batchlense turns that into a repeatable workflow: **calibrate** what “normal” looks like, **score** new batches the same way every time, and **see** which features drove a flag—without asking analysts to run notebooks or maintain ML infra.

---

## Features

- **Demo mode** — synthetic chemical-style batches with injected outliers on first load  
- **LOF anomaly detection** — reference-calibrated threshold; sklearn backend by default, optional PyTorch path  
- **SPC-style views** — timelines, batch comparison vs median, radar (z-scores)  
- **Process capability** — **Pp / Ppk** from optional spec file (`feature`, `usl`, `lsl`)  
- **2D maps** — PCA, t-SNE, UMAP for structure at a glance  
- **Exports** — scored CSV and PDF summary (Plotly + Kaleido)

---

## Quick start

```bash
git clone <your-repo-url> && cd batchlense
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
batchlense
```

Open <http://127.0.0.1:8501>. Optional GPU stack: `pip install -e ".[gpu]"` (installs PyTorch for the legacy tensor LOF path).

---

## Tech stack

Python **3.11+**, Pandas, NumPy, **scikit-learn** (LOF + projections), Streamlit, Plotly, SciPy, UMAP, fpdf2, Kaleido. **PyTorch is optional** (`[gpu]` extra): the app and CLI use sklearn LOF and NumPy normalization when torch is not installed (e.g. Streamlit Cloud).

---

## How it works

1. **Upload** batch CSV (and optionally a clean **reference** CSV plus **spec limits**).  
2. **Configure** numeric feature columns and LOF settings (`k`, normalization, threshold σ).  
3. **Normalize** features (z-score, min–max, or none) using **reference** statistics when a reference set is provided.  
4. **Score** each input row with LOF; rows above the calibrated threshold are **flagged**.  
5. **Explore** KPIs, charts, correlation heatmaps, and exports.

---

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format src tests
mypy
pytest
```

---

## Troubleshooting

<details>
<summary><strong>Expand</strong> — Kaleido, optional torch, zsh, LAN binding</summary>

**PDF / static charts:** This repo pins **`kaleido==0.2.1`** so Plotly can render PNGs without Chrome. Newer **`0.2.1.post1`** often lacks manylinux **x86_64** wheels (common **`uv`** failure). Kaleido **1.x** needs Chrome or `plotly_get_chrome`.

**Optional PyTorch:** Core installs do not include torch. If you want the original tensor LOF path, use `pip install -e ".[gpu]"`. If you see `ModuleNotFoundError: torch`, either install the extra or ignore it—sklearn LOF runs without it. If torch *is* installed but import fails, confirm you’re using the same venv you installed into (`python -c "import torch; print(torch.__file__)"`).

**zsh:** Prefer `python -m streamlit run src/frontend/app.py` so zsh doesn’t “correct” `streamlit` to `.streamlit`.

**Listen on LAN:** `batchlense --host 0.0.0.0 --port 8501` (default is loopback-only via launcher + `.streamlit/config.toml`).

</details>
