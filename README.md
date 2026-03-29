# batchlens

A lightweight tool for production and quality teams to detect anomalies 
in batch telemetry data. Upload your batch export, and batchlens flags 
which batches deviate from historical baselines across process parameters 
like heat, rotation, grind, or any other recorded metric.

Built for teams who need answers fast, without a data science background.

---

## What it does

- Upload Excel batch data through a simple web interface
- Compare new batches against historical production data
- Detect deviations in process telemetry automatically
- Flag batches worth a second look before they become a problem

---

## Screenshots

> Demo and screenshots coming soon.

---

## Run it locally

**Requirements**
- Python 3.10+
- pip

**Setup**

git clone https://github.com/404rorre/batchlens.git
cd batchlens
pip install -r requirements.txt
streamlit run app.py

Then open your browser at http://localhost:8501.

---

## Tech stack

- Streamlit
- Pandas
- Python
