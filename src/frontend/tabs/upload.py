"""Upload CSVs, template download, NaN handling."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TEMPLATE_PATH = ASSETS / "template.csv"


def render_upload_tab() -> None:
    st.header("Upload data")
    st.markdown(
        "Upload your **production batch** CSV. "
        "Optionally add a **reference** file of known good batches — this improves "
        "how the tool sets the outlier threshold. If you skip it, the same file is "
        "used for both calibration and scoring (less ideal but still works).",
    )

    main = st.file_uploader(
        "Production data CSV (required)", type=["csv"], key="main_csv"
    )
    ref = st.file_uploader(
        "Reference CSV (optional, known good quality)",
        type=["csv"],
        key="ref_csv",
    )
    spec = st.file_uploader(
        "Specification limits CSV (optional, for Cp/Cpk) — columns: feature, usl, lsl",
        type=["csv"],
        key="spec_csv",
    )

    if TEMPLATE_PATH.is_file():
        st.download_button(
            label="Download example CSV template",
            data=TEMPLATE_PATH.read_bytes(),
            file_name="batchlense_template.csv",
            mime="text/csv",
        )

    if main is None:
        st.info("Upload a CSV to continue.")
        st.session_state.pop("df_main", None)
        return

    df = pd.read_csv(main)
    st.session_state["df_main"] = df
    st.session_state["ref_bytes"] = ref.getvalue() if ref else None
    st.session_state["spec_bytes"] = spec.getvalue() if spec else None

    st.subheader("Preview")
    st.dataframe(df.head(10), width="stretch")

    nan_count = df.isna().sum().sum()
    if nan_count:
        st.warning(f"This file has **{nan_count}** missing cells.")
        mode = st.radio(
            "How should we handle missing values?",
            ["Drop rows with any NaN", "Fill numeric with column median"],
            horizontal=True,
            key="nan_mode",
        )
        if mode.startswith("Drop"):
            df2 = df.dropna()
            st.session_state["df_main"] = df2
            st.success(f"Using {len(df2)} rows after dropping NaNs (was {len(df)}).")
        else:
            num = df.select_dtypes(include=["number"]).columns
            df2 = df.copy()
            for c in num:
                df2[c] = df2[c].fillna(df2[c].median())
            df2 = df2.dropna()
            st.session_state["df_main"] = df2
            st.success(
                "Filled numeric NaNs with medians and dropped remaining empty rows."
            )

    if st.session_state.get("ref_bytes"):
        ref_df = pd.read_csv(io.BytesIO(st.session_state["ref_bytes"]))
        st.session_state["df_ref"] = ref_df
        st.caption(f"Reference file loaded: **{len(ref_df)}** rows.")
    else:
        st.session_state.pop("df_ref", None)

    if st.session_state.get("spec_bytes"):
        spec_df = pd.read_csv(io.BytesIO(st.session_state["spec_bytes"]))
        st.session_state["df_spec"] = spec_df
        st.caption("Specification limits loaded for capability metrics.")
    else:
        st.session_state.pop("df_spec", None)
