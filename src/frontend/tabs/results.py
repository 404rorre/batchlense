"""Run LOF, charts, KPIs, and QC views."""

from __future__ import annotations

import hashlib
import io
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.helper_modules.dataloader import DataLoader, Normalization
from backend.lof.pipeline import run_lof_on_dataframe
from frontend.components.batch_comparison import comparison_table, make_delta_bar_figure
from frontend.components.batch_timeline import make_timeline_figure
from frontend.components.correlation_heatmap import make_correlation_heatmap_figure
from frontend.components.dim_reduction import Method, project_2d
from frontend.components.feature_histogram import make_feature_histogram_figure
from frontend.components.kpi import basic_kpis, compute_capability, cpk_status
from frontend.components.radar_chart import make_radar_figure, z_scores_for_batch


def _plotly_template() -> str:
    try:
        if getattr(st.context, "theme", None) and st.context.theme.type == "dark":
            return "plotly_dark"
    except Exception:
        pass
    return "plotly_white"


def _fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()


@st.cache_data(show_spinner="Computing LOF scores…")
def _cached_lof(
    fp_main: str,
    fp_ref: str | None,
    features_key: str,
    norm: str,
    n_neighbors: int,
    sigma: float,
    main_csv: bytes,
    ref_csv: bytes | None,
) -> tuple[bytes, float, tuple[str, ...]]:
    """Serialize result as CSV bytes + threshold + warnings tuple."""
    df = pd.read_csv(io.BytesIO(main_csv))
    ref_df = pd.read_csv(io.BytesIO(ref_csv)) if ref_csv else None
    features = tuple(features_key.split("||"))
    out, thr, warns = run_lof_on_dataframe(
        df,
        list(features),
        reference_df=ref_df,
        normalization=cast(Normalization, norm),
        n_neighbors=n_neighbors,
        threshold_sigma=sigma,
    )
    bio = pd.io.common.BytesIO()
    out.to_csv(bio, index=False)
    return bio.getvalue(), thr, tuple(warns)


def render_results_tab() -> None:
    st.header("Results")
    df = st.session_state.get("df_main")
    feats = st.session_state.get("feature_cols")
    if df is None or not feats:
        st.warning("Complete **Upload** and **Configure** first.")
        return
    if len(df) < 2:
        st.error("LOF needs **at least 2 rows** in the uploaded data.")
        return

    norm = cast(Normalization, st.session_state.get("normalization", "zscore"))
    n_neighbors = st.session_state.get("n_neighbors", 20)
    sigma = st.session_state.get("threshold_sigma", 3.0)
    batch_col = st.session_state.get("batch_col")
    dt_col = st.session_state.get("datetime_col")
    dr_method = st.session_state.get("dim_reduction", "pca")

    main_csv = df.to_csv(index=False).encode()
    ref_df = st.session_state.get("df_ref")
    ref_csv = ref_df.to_csv(index=False).encode() if ref_df is not None else None
    fp_main = _fingerprint(df)
    fp_ref = _fingerprint(ref_df) if ref_df is not None else ""

    if st.button("Run analysis", type="primary", key="run_lof_btn"):
        try:
            blob, thr, warns = _cached_lof(
                fp_main,
                fp_ref,
                "||".join(feats),
                str(norm),
                n_neighbors,
                sigma,
                main_csv,
                ref_csv,
            )
            scored = pd.read_csv(io.BytesIO(blob))
            st.session_state["scored_df"] = scored
            st.session_state["lof_threshold"] = thr
            st.session_state["lof_warnings"] = list(warns)
        except Exception as e:
            st.error(f"LOF failed: {e}")
            return

    scored = st.session_state.get("scored_df")
    if scored is None:
        st.info("Click **Run analysis** to score your batches.")
        return

    for w in st.session_state.get("lof_warnings", []):
        st.warning(w)
    st.caption(
        f"Calibrated LOF threshold: **{st.session_state.get('lof_threshold', 0):.4f}**"
    )

    tpl = _plotly_template()
    loader = DataLoader(normalization=norm)
    X_t = loader.fit(data=scored[feats].astype(float).values.tolist())
    X_np = X_t.detach().cpu().numpy()
    coords, dr_warns = project_2d(X_np, cast(Method, dr_method))
    for w in dr_warns:
        st.caption(w)

    is_out = scored["is_outlier"].astype(int).values
    colors = np.where(is_out == 1, "#FF8C00", "#D3D3D3")
    hover = []
    for i in range(len(scored)):
        parts = []
        if batch_col and batch_col in scored.columns:
            parts.append(f"Batch: {scored.iloc[i][batch_col]}")
        if dt_col and dt_col in scored.columns:
            parts.append(f"Date: {scored.iloc[i][dt_col]}")
        parts.append(f"LOF: {scored.iloc[i]['lof_score']:.4f}")
        for f in feats[:5]:
            parts.append(f"{f}: {scored.iloc[i][f]}")
        hover.append("<br>".join(parts))

    fig_scatter = go.Figure(
        data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(size=10, color=colors, line=dict(width=0.5, color="white")),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
    )
    fig_scatter.update_layout(
        title=f"Batch map ({dr_method.upper()}) — orange = review",
        template=tpl,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        height=520,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.session_state["fig_scatter"] = fig_scatter

    kpis = basic_kpis(scored, feats)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batches", kpis["n_batches"])
    m2.metric("Outliers", kpis["n_outliers"])
    m3.metric("Outlier rate", f"{kpis['outlier_rate_pct']:.1f}%")
    m4.metric("LOF range", f"{kpis['lof_min']:.2f} – {kpis['lof_max']:.2f}")
    st.session_state["kpis_dict"] = kpis

    spec_df = st.session_state.get("df_spec")
    cap = compute_capability(scored, feats, spec_df) if spec_df is not None else None
    if cap is not None and not cap.empty:
        st.subheader("Process capability (Cp / Cpk)")
        st.caption("Uses overall variation; compare to your internal SPC rules.")
        disp = cap.copy()
        disp["status"] = disp["cpk"].apply(cpk_status)
        st.dataframe(disp, use_container_width=True)

    st.subheader("Correlation between parameters")
    st.caption(
        "Strong color = parameters move together (or opposite). "
        "Blocks show groups that tend to change as a group.",
    )
    fig_corr = make_correlation_heatmap_figure(scored, feats, template=tpl)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Outlier profile (radar)")
    out_idx = scored.index[scored["is_outlier"] == 1].tolist()
    if out_idx and batch_col:
        labels = scored.loc[out_idx, batch_col].astype(str).tolist()
        pick = st.selectbox("Pick an outlier batch", options=labels, key="radar_pick")
        mask = scored[batch_col].astype(str) == pick
        _, z_b = z_scores_for_batch(scored, feats, mask)
        z_ref = np.zeros(len(feats))
        fig_r = make_radar_figure(
            feats, z_ref, z_b, template=tpl, title=f"Batch {pick} vs average"
        )
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.caption(
            "Run analysis and flag outliers to see a radar chart for a specific batch."
        )

    st.subheader("Distributions per feature")
    hf = st.selectbox("Feature for histogram", options=feats, key="hist_feat")
    fig_h = make_feature_histogram_figure(scored, hf, template=tpl)
    st.plotly_chart(fig_h, use_container_width=True)

    if dt_col:
        st.subheader("Trend over time")
        res = st.radio(
            "Time grouping", ["day", "month", "hour"], horizontal=True, key="ts_res"
        )
        tf = st.selectbox("Feature for timeline", options=feats, key="tl_feat")
        fig_tl = make_timeline_figure(
            scored,
            datetime_col=dt_col,
            feature=tf,
            batch_col=batch_col,
            resample=res,  # type: ignore[arg-type]
            template=tpl,
        )
        st.plotly_chart(fig_tl, use_container_width=True)
        st.session_state["fig_timeline"] = fig_tl
    else:
        st.session_state.pop("fig_timeline", None)

    st.subheader("Compare batch to typical (median)")
    if batch_col:
        b_pick = st.selectbox(
            "Batch to compare",
            options=scored[batch_col].astype(str).unique().tolist(),
            key="cmp_pick",
        )
        comp = comparison_table(scored, feats, batch_col, b_pick)
        if not comp.empty:
            st.dataframe(comp, use_container_width=True)
            fig_b = make_delta_bar_figure(comp, template=tpl)
            st.plotly_chart(fig_b, use_container_width=True)

    st.info(
        "**What LOF means:** each batch gets a score. "
        "Around **1** = similar to neighbors; **well above 1** = unusually isolated — worth a look. "
        "Orange points are above the automatic threshold from your reference data (or from the same file if no reference). "
        "Use the radar and comparison tools to see *which* parameters look off before changing the process.",
    )
