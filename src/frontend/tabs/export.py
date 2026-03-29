"""Download scored CSV, outlier-only CSV, and PDF report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from frontend.components.pdf_report import build_pdf


def _export_figure_png(fig: object, *, width: int, height: int) -> Path | None:
    """Write Plotly figure to a temp PNG; return path or None on failure."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    p = Path(tmp.name)
    try:
        write_image = getattr(fig, "write_image", None)
        if write_image is None:
            return None
        write_image(str(p), width=width, height=height, scale=1)
        return p
    except Exception:
        p.unlink(missing_ok=True)
        return None


def render_export_tab() -> None:
    st.header("Export")
    scored = st.session_state.get("scored_df")
    if scored is None:
        st.warning("Run **Results** analysis first.")
        return

    feats = st.session_state.get("feature_cols", [])
    batch_col = st.session_state.get("batch_col")
    dt_col = st.session_state.get("datetime_col")
    kpis = st.session_state.get("kpis_dict", {})

    full_csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download full scored CSV (original order + LOF + flag)",
        data=full_csv,
        file_name="batchlense_scored.csv",
        mime="text/csv",
    )

    only_out = scored[scored["is_outlier"] == 1]
    st.download_button(
        label="Download outlier rows only (for follow-up)",
        data=only_out.to_csv(index=False).encode("utf-8"),
        file_name="batchlense_outliers_only.csv",
        mime="text/csv",
    )

    if st.button("Build PDF report", key="build_pdf"):
        paths: list[Path | None] = []
        fig_s = st.session_state.get("fig_scatter")
        if fig_s is not None:
            p = _export_figure_png(fig_s, width=1100, height=600)
            if p is None:
                st.caption(
                    "Scatter plot could not be exported (install **kaleido** or check Plotly)."
                )
            paths.append(p)

        fig_t = st.session_state.get("fig_timeline")
        if fig_t is not None:
            p2 = _export_figure_png(fig_t, width=1100, height=400)
            if p2 is None:
                st.caption(
                    "Timeline could not be exported (install **kaleido** or check Plotly)."
                )
            paths.append(p2)

        try:
            pdf_bytes = build_pdf(
                kpis=kpis,
                scored_df=scored,
                feature_cols=feats,
                batch_col=batch_col,
                datetime_col=dt_col,
                image_paths=paths,
            )
            st.session_state["pdf_bytes"] = pdf_bytes
            st.success("PDF ready — use the download button below.")
        finally:
            for p in paths:
                if p is not None and p.is_file():
                    p.unlink(missing_ok=True)

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            label="Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name="batchlense_report.pdf",
            mime="application/pdf",
        )
