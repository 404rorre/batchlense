"""Generate PDF summary report with fpdf2."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

LINKEDIN = "https://www.linkedin.com/in/mwollenb/"


def _fpdf_text(s: str) -> str:
    """Core PDF fonts only support Latin-1; drop/replace other characters."""
    return s.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(
    *,
    kpis: dict[str, float | int],
    scored_df: pd.DataFrame,
    feature_cols: list[str],
    batch_col: str | None,
    datetime_col: str | None,
    image_paths: list[Path | None],
) -> bytes:
    """Return PDF bytes. ``image_paths`` optional PNG files from plotly/kaleido."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        10,
        _fpdf_text("batchlense - production quality report"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _fpdf_text("Max Wollenberg"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 200)
    pdf.cell(0, 8, _fpdf_text(LINKEDIN), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _fpdf_text("Summary KPIs"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        _fpdf_text(f"Total batches: {kpis.get('n_batches', 0)}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        6,
        _fpdf_text(f"Outliers: {kpis.get('n_outliers', 0)}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        6,
        _fpdf_text(f"Outlier rate: {kpis.get('outlier_rate_pct', 0):.2f}%"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.cell(
        0,
        6,
        _fpdf_text(
            f"LOF score range: {kpis.get('lof_min', 0):.4f} - {kpis.get('lof_max', 0):.4f}",
        ),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _fpdf_text("Outlier list"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    sub = scored_df[scored_df["is_outlier"] == 1]
    cols = [c for c in [batch_col, datetime_col, "lof_score"] if c and c in sub.columns]
    if not cols:
        cols = ["lof_score"]
    for _, row in sub[cols].head(50).iterrows():
        line = " | ".join(f"{c}={row[c]}" for c in cols)
        pdf.multi_cell(
            0,
            5,
            _fpdf_text(line),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    if len(sub) > 50:
        pdf.cell(
            0,
            6,
            _fpdf_text(f"... and {len(sub) - 50} more rows (see CSV export)."),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    for p in image_paths:
        if p is None or not Path(p).is_file():
            continue
        pdf.add_page()
        pdf.image(str(p), x=10, y=20, w=190)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
