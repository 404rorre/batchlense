"""Column mapping and algorithm parameters."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _datetime_candidates(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().sum() >= max(1, len(df) // 2):
            out.append(c)
    return out


def render_configure_tab() -> None:
    st.header("Configure analysis")
    df = st.session_state.get("df_main")
    if df is None:
        st.warning("Upload a CSV in the **Upload** tab first.")
        return

    if len(df) < 2:
        st.error("LOF needs **at least 2 data rows**. Add more batches or merge files.")
        return

    num_cols = _numeric_columns(df)
    if not num_cols:
        st.error(
            "No numeric columns found. Add numeric process parameters to your CSV."
        )
        return

    st.subheader("Feature columns (LOF uses these)")
    chosen: list[str] = []
    cols = st.columns(3)
    for i, c in enumerate(num_cols):
        with cols[i % 3]:
            if st.checkbox(c, value=True, key=f"feat_{c}"):
                chosen.append(c)

    if not chosen:
        st.warning("No features selected.")
        if st.button("Use all numeric columns (auto-detect)", key="auto_feat"):
            st.session_state["feature_cols"] = num_cols
            st.rerun()
        return

    st.session_state["feature_cols"] = chosen

    all_cols = list(df.columns)
    st.subheader("Batch identifier")
    default_b = 0
    if "batch_number" in all_cols:
        default_b = all_cols.index("batch_number")
    batch_col = st.selectbox(
        "Column that identifies each batch",
        options=all_cols,
        index=default_b,
        help="Used in charts, hover text, and exports. Often named batch_number or lot_id.",
    )
    st.session_state["batch_col"] = batch_col

    dt_cands = _datetime_candidates(df)
    if len(dt_cands) > 1:
        st.info(
            f"Several columns look like dates: {', '.join(dt_cands)}. Pick one for time charts."
        )
    st.subheader("Production date (optional)")
    dt_options = ["(none)"] + dt_cands
    dt_pick = st.selectbox(
        "Datetime column for trends",
        options=dt_options,
        help="If set, you get a timeline and time-based outlier view.",
    )
    st.session_state["datetime_col"] = None if dt_pick == "(none)" else dt_pick

    st.subheader("Local Outlier Factor (LOF)")
    c1, c2, c3 = st.columns(3)
    max_k = max(2, min(50, len(df) - 1)) if len(df) > 2 else 2
    default_k = min(20, max_k)
    with c1:
        n_neighbors = st.slider(
            "Number of neighbors (k)",
            min_value=2,
            max_value=max_k,
            value=default_k,
            help=(
                "How many nearby batches define 'normal' for each point. "
                "Standard default is 20. Lower = more sensitive; higher = smoother."
            ),
        )
    with c2:
        norm = st.radio(
            "Feature scaling",
            ["zscore", "minmax", "none"],
            horizontal=True,
            help=(
                "z-score: each column mean 0, std 1 (good when units differ). "
                "min-max: scale to 0–1. none: raw values."
            ),
        )
    with c3:
        sigma = st.slider(
            "Threshold strictness (σ)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help=(
                "Threshold = average LOF on reference + σ × spread. "
                "Higher σ = fewer points flagged as outliers."
            ),
        )

    st.session_state["n_neighbors"] = n_neighbors
    st.session_state["normalization"] = norm
    st.session_state["threshold_sigma"] = sigma

    st.subheader("2D map (how points are drawn)")
    dr = st.radio(
        "Projection method",
        ["pca", "tsne", "umap"],
        horizontal=True,
        help=(
            "**PCA** — big-picture global structure. "
            "**t-SNE** — tight local clusters. "
            "**UMAP** — keeps both local neighborhoods and broader layout. "
            "All use a fixed random seed so the map looks the same each run."
        ),
    )
    st.session_state["dim_reduction"] = dr
