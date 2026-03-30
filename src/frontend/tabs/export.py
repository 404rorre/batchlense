"""Download scored CSV, outlier-only CSV, and PDF report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from frontend.components.pdf_report import build_pdf


def _export_figure_png(
    fig: object,
    *,
    width: int,
    height: int,
) -> tuple[Path | None, str | None]:
    """Write Plotly figure to a temp PNG. Returns ``(path, None)`` or ``(None, error)``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    p = Path(tmp.name)
    try:
        write_image = getattr(fig, "write_image", None)
        if write_image is None:
            p.unlink(missing_ok=True)
            return None, "figure has no write_image"
        write_image(str(p), width=width, height=height, scale=1, engine="kaleido")
        return p, None
    except Exception as e:
        p.unlink(missing_ok=True)
        msg = str(e).strip().split("\n")[0]
        if len(msg) > 240:
            msg = msg[:237] + "..."
        return None, msg


def render_export_section(
    *,
    key_prefix: str,
    show_header: bool = True,
) -> None:
    """
    CSV + PDF export widgets. ``key_prefix`` keeps Streamlit keys unique when
    this block appears in multiple places (e.g. Results bottom + Export tab).
    """
    scored = st.session_state.get("scored_df")
    if scored is None:
        st.warning("Run **Results** analysis first.")
        return

    feats = st.session_state.get("feature_cols", [])
    batch_col = st.session_state.get("batch_col")
    dt_col = st.session_state.get("datetime_col")
    kpis = st.session_state.get("kpis_dict", {})

    if show_header:
        st.subheader("Export")

    full_csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download full scored CSV (original order + LOF + flag)",
        data=full_csv,
        file_name="batchlense_scored.csv",
        mime="text/csv",
        key=f"{key_prefix}dl_full",
    )

    only_out = scored[scored["is_outlier"] == 1]
    st.download_button(
        label="Download outlier rows only (for follow-up)",
        data=only_out.to_csv(index=False).encode("utf-8"),
        file_name="batchlense_outliers_only.csv",
        mime="text/csv",
        key=f"{key_prefix}dl_out",
    )

    if st.button("Build PDF report", key=f"{key_prefix}build_pdf"):
        paths: list[Path | None] = []
        fig_s = st.session_state.get("fig_scatter")
        if fig_s is not None:
            p, err = _export_figure_png(fig_s, width=1100, height=600)
            if p is None:
                st.caption(
                    f"Scatter plot could not be exported: {err or 'unknown error'}. "
                    "Project pins **kaleido==0.2.1** (avoids broken **0.2.1.post1** on Linux x86_64). "
                    "Or use **kaleido>=1** with Chrome / `plotly_get_chrome`.",
                )
            paths.append(p)

        fig_t = st.session_state.get("fig_timeline")
        if fig_t is not None:
            p2, err2 = _export_figure_png(fig_t, width=1100, height=400)
            if p2 is None:
                st.caption(
                    f"Timeline could not be exported: {err2 or 'unknown error'}. "
                    "Same as scatter: **kaleido==0.2.1** or Chrome + kaleido 1.x.",
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
            st.session_state[f"{key_prefix}pdf_bytes"] = pdf_bytes
            st.success("PDF ready — use the download button below.")
        finally:
            for p in paths:
                if p is not None and p.is_file():
                    p.unlink(missing_ok=True)

    pdf_key = f"{key_prefix}pdf_bytes"
    if st.session_state.get(pdf_key):
        st.download_button(
            label="Download PDF report",
            data=st.session_state[pdf_key],
            file_name="batchlense_report.pdf",
            mime="application/pdf",
            key=f"{key_prefix}dl_pdf",
        )


def render_export_tab() -> None:
    st.header("Export")
    render_export_section(key_prefix="tab_export_", show_header=False)
