"""Streamlit entry point: batchlense LOF dashboard."""

from __future__ import annotations

import streamlit as st

from frontend.tabs.configure import render_configure_tab
from frontend.tabs.export import render_export_tab
from frontend.tabs.results import render_results_tab
from frontend.tabs.upload import render_upload_tab


def main() -> None:
    st.set_page_config(
        page_title="batchlense",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("batchlense")
    st.caption(
        "Local Outlier Factor (LOF) for batch production telemetry — upload, configure, review, export."
    )

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
