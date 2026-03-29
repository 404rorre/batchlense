# batchlense

A lightweight tool for production and quality teams to detect anomalies in batch telemetry data. Upload batch exports and **batchlense** flags which batches deviate from historical baselines across process parameters (temperature, pressure, rotation, grind force, or any numeric metrics).

Built for teams who need answers fast, without a data science background.

---

## What it does

- **Web UI (Streamlit):** Upload CSV, pick features and LOF settings, explore PCA / t-SNE / UMAP maps, KPIs, correlation heatmaps, radar and distribution views, optional Cpk/Ppk from a spec-limits file, then export scored CSV and a PDF summary.
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
streamlit run src/frontend/app.py
```

Open <http://localhost:8501>.

If you run without installing the package, add `src` to `PYTHONPATH`:

```bash
PYTHONPATH=src streamlit run src/frontend/app.py
```

Optional **reference CSV** and **spec limits** (`feature`, `usl`, `lsl`) uploaders are described in the app. A sample template is available from the Upload tab (`batch_number`, `production_date`, plus numeric columns).

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
