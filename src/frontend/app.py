"""Streamlit entry point: batchlense LOF dashboard."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from frontend.mock_data import FEATURE_COLUMNS, generate_mock_batches
from frontend.tabs.configure import render_configure_tab
from frontend.tabs.export import render_export_tab
from frontend.tabs.results import _cached_lof, _fingerprint, render_results_tab
from frontend.tabs.upload import render_upload_tab


def _seed_demo_session() -> None:
    df, _mask = generate_mock_batches()
    feats = list(FEATURE_COLUMNS)
    n_rows = len(df)
    max_k = max(2, min(50, n_rows - 1)) if n_rows > 2 else 2
    k = min(20, max_k)

    st.session_state["df_main"] = df
    st.session_state["feature_cols"] = feats
    st.session_state["batch_col"] = "batch_number"
    st.session_state["datetime_col"] = "production_date"
    st.session_state["n_neighbors"] = k
    st.session_state["normalization"] = "zscore"
    st.session_state["threshold_sigma"] = 3.0
    st.session_state["dim_reduction"] = "pca"
    st.session_state.pop("df_ref", None)
    st.session_state.pop("ref_bytes", None)
    st.session_state.pop("df_spec", None)
    st.session_state.pop("spec_bytes", None)

    main_csv = df.to_csv(index=False).encode()
    fp_main = _fingerprint(df)
    blob, thr, warns = _cached_lof(
        fp_main,
        "",
        "||".join(feats),
        "zscore",
        k,
        3.0,
        main_csv,
        None,
    )
    scored = pd.read_csv(io.BytesIO(blob))
    st.session_state["scored_df"] = scored
    st.session_state["lof_threshold"] = thr
    st.session_state["lof_warnings"] = list(warns)
    st.session_state["demo_seeded"] = True


def _clear_demo_and_switch_to_upload() -> None:
    for key in (
        "df_main",
        "scored_df",
        "lof_threshold",
        "lof_warnings",
        "fig_scatter",
        "fig_timeline",
        "pdf_bytes",
        "kpis_dict",
        "feature_cols",
        "batch_col",
        "datetime_col",
        "df_ref",
        "df_spec",
        "ref_bytes",
        "spec_bytes",
        "demo_seeded",
    ):
        st.session_state.pop(key, None)
    st.session_state["mode"] = "upload"


def main() -> None:
    st.set_page_config(
        page_title="batchlense",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "mode" not in st.session_state:
        st.session_state["mode"] = "demo"

    st.title("batchlense")
    st.caption(
        "Local Outlier Factor (LOF) for batch production telemetry — upload, configure, review, export."
    )

    if st.session_state["mode"] == "demo":
        if not st.session_state.get("demo_seeded"):
            _seed_demo_session()

        st.success(
            "**Demo mode** — showing synthetic chemical batch data (reactor temperature, "
            "pressure, pH, viscosity, catalyst, yield) over one week. "
            "Outliers are injected in the mock generator; LOF was run automatically."
        )
        render_results_tab()

        st.divider()
        if st.button(
            "Feed your own data",
            type="primary",
            width="stretch",
            key="switch_to_upload",
        ):
            _clear_demo_and_switch_to_upload()
            st.rerun()
        return

    tab_u, tab_c, tab_r, tab_e = st.tabs(
        ["1 · Upload", "2 · Configure", "3 · Results", "4 · Export"],
    )
    with tab_u:
        render_upload_tab()
    with tab_c:
        render_configure_tab()
    with tab_r:
        render_results_tab()
    with tab_e:
        render_export_tab()


main()
