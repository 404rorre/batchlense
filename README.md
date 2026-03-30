# batchlense

A lightweight tool for production and quality teams to detect anomalies in batch telemetry data. Upload batch exports and **batchlense** flags which batches deviate from historical baselines across process parameters (temperature, pressure, rotation, grind force, or any numeric metrics).

Built for teams who need answers fast, without a data science background.

---

## What it does

- **Web UI (Streamlit):** On first open you land in **demo mode** with synthetic chemical batch data (one week of timestamps, injected outliers) and LOF already run. Use **Feed your own data** to switch to the full flow: upload CSV, configure features and LOF, explore PCA / t-SNE / UMAP maps, KPIs, correlation heatmaps, radar and distribution views, optional Cpk/Ppk from a spec-limits file, then export scored CSV and a PDF summary.
- **CLI:** Score batches from the command line using the same LOF pipeline.

---

## Run the Streamlit app locally

**Requirements:** Python 3.11+

```bash
git clone https://github.com/404rorre/batchlens.git   # or your fork
cd batchlense   # use your clone directory name
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
batchlense
```

This starts Streamlit on **127.0.0.1:8501** (IPv4 loopback only — not reachable from other machines). Open <http://127.0.0.1:8501> or <http://localhost:8501>.

On some systems, `localhost` makes Streamlit listen on `[::]:8501` (all interfaces). This project defaults to **127.0.0.1** via the `batchlense` launcher and [`.streamlit/config.toml`](.streamlit/config.toml) when you run from the repo root.

To listen on all interfaces (e.g. share on your LAN):

```bash
batchlense --host 0.0.0.0 --port 8501
```

You can also run Streamlit directly (from the **repository root** so `.streamlit/config.toml` applies). Prefer **`python -m streamlit`** so zsh’s spell-checker does not prompt to “correct” `streamlit` to `.streamlit` (the config folder):

```bash
python -m streamlit run src/frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

If you run without installing the package, add `src` to `PYTHONPATH`:

```bash
PYTHONPATH=src python -m streamlit run src/frontend/app.py --server.address 127.0.0.1
```

Optional **reference CSV** and **spec limits** (`feature`, `usl`, `lsl`) uploaders are described in the app. A sample template is available from the Upload tab (`batch_number`, `production_date`, plus numeric columns).

**PDF report charts:** Dependencies pin **`kaleido==0.2.1`** so Plotly can export PNGs without Chrome. Newer **0.2.1.post1** lacks manylinux **x86_64** wheels (common **`uv`** failure); **1.x** needs Chrome or **`plotly_get_chrome`**.

**`ModuleNotFoundError: No module named 'torch'`:** Dependencies are installed into **one** environment (e.g. `.venv`), but Streamlit was started with **another** Python (e.g. pyenv global). Fix: **activate the same venv** you used for `pip install` / `uv pip install`, then start the app:

```bash
source .venv/bin/activate   # must be this shell before streamlit
python -c "import torch; print(torch.__file__)"   # should point under .venv
python -m streamlit run src/frontend/app.py --server.address 127.0.0.1
```

With **uv**, you can avoid activation mistakes:

```bash
uv pip install -e ".[dev]"
uv run python -m streamlit run src/frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

---

## CLI usage

After `pip install -e .`:

```bash
batchlense-lof --help
```

Configure input/reference batch CSV paths and LOF options as documented by `--help`.

---

## Screenshots

> Demo and screenshots coming soon.

---

## Tech stack

- Python, Pandas, NumPy  
- PyTorch (normalization / tensor path in `DataLoader` + `LOF`)  
- scikit-learn (LOF internals)  
- Streamlit, Plotly, UMAP  
- fpdf2 + Kaleido (PDF + static chart export)

---

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format src tests
mypy
pytest
```
