"""Run LOF, charts, KPIs, and QC views."""

from __future__ import annotations

import hashlib
import io
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.helper_modules.numpy_normalize import (
    Normalization,
    fit_transform_features_numpy,
)
from backend.lof.pipeline import HAS_TORCH, run_lof_on_dataframe
from frontend.components.batch_comparison import comparison_table, make_delta_bar_figure
from frontend.components.batch_timeline import (
    baseline_feature_series,
    has_in_control_baseline,
    make_production_bar_figure,
    make_timeline_figure,
)
from frontend.components.correlation_heatmap import make_correlation_heatmap_figure
from frontend.components.dim_reduction import Method, project_2d
from frontend.components.feature_histogram import make_feature_histogram_figure
from frontend.components.kpi import (
    auto_spec_limits,
    basic_kpis,
    compute_capability,
    ppk_status,
)
from frontend.components.radar_chart import (
    RADAR_DISPLAY_MAX,
    RADAR_FIGURE_HEIGHT,
    make_radar_figure,
    subset_z_for_features,
    z_scores_for_batch,
)
from frontend.components.zscore_grid import make_zscore_grid_figure
from frontend.tabs.export import render_export_section


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


def _top_problem_features(
    scored: pd.DataFrame, feats: list[str]
) -> list[tuple[str, float]]:
    """Mean |z| vs population among outlier rows only; return top 3 by that score."""
    outs = scored[scored["is_outlier"] == 1]
    if outs.empty:
        return []
    pop = scored[feats].astype(float)
    mean = pop.mean()
    std = pop.std(ddof=1)
    ranked: list[tuple[str, float]] = []
    for c in feats:
        s = float(std.get(c, np.nan)) if c in std.index else float("nan")
        if not np.isfinite(s) or s <= 0:
            continue
        z_mean = float(((outs[c].astype(float) - mean[c]) / s).abs().mean())
        if np.isfinite(z_mean):
            ranked.append((c, z_mean))
    ranked.sort(key=lambda x: -x[1])
    return ranked[:3]


def _lookup_spec_limits(
    spec_df: pd.DataFrame | None,
    feature: str,
) -> tuple[float | None, float | None]:
    """Return (USL, LSL) for ``feature`` from a spec table (feature / usl / lsl columns)."""
    if spec_df is None or spec_df.empty:
        return None, None
    s = spec_df.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    if not {"feature", "usl", "lsl"}.issubset(s.columns):
        return None, None
    match = s[s["feature"].astype(str).str.strip() == str(feature).strip()]
    if match.empty:
        return None, None
    try:
        usl = float(match.iloc[0]["usl"])
        lsl = float(match.iloc[0]["lsl"])
    except (ValueError, TypeError):
        return None, None
    if not (np.isfinite(usl) and np.isfinite(lsl)) or usl <= lsl:
        return None, None
    return usl, lsl


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
    kpis = basic_kpis(scored, feats)
    st.session_state["kpis_dict"] = kpis

    st.subheader("Executive summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Batches", kpis["n_batches"])
    m2.metric("Outliers", kpis["n_outliers"])
    m3.metric("Outlier rate", f"{kpis['outlier_rate_pct']:.1f}%")
    m4.metric("LOF range", f"{kpis['lof_min']:.2f} – {kpis['lof_max']:.2f}")

    top_prob = _top_problem_features(scored, feats)
    if top_prob:
        st.caption(
            "Features that deviate most in **flagged** batches (mean |z| vs all data)."
        )
        cols = st.columns(min(3, len(top_prob)))
        for col, (name, zavg) in zip(cols, top_prob, strict=True):
            with col:
                st.metric(
                    label=name.replace("_", " "),
                    value=f"{zavg:.2f}",
                    help="|z| vs population mean",
                )
    else:
        st.caption("No outlier batches to rank — run analysis or relax the threshold.")

    current_dr = st.session_state.get("dim_reduction", "pca")
    if current_dr not in ("pca", "tsne", "umap"):
        current_dr = "pca"
    dr_method = st.radio(
        "Projection (2D map)",
        ["pca", "tsne", "umap"],
        horizontal=True,
        key="dr_method_results",
        index=["pca", "tsne", "umap"].index(current_dr),
        help="PCA: global structure. t-SNE: local clusters. UMAP: both.",
    )
    st.session_state["dim_reduction"] = dr_method

    if HAS_TORCH:
        from backend.helper_modules.dataloader import DataLoader

        loader = DataLoader(normalization=norm)
        X_t = loader.fit(data=scored[feats].astype(float).values.tolist())
        X_np = X_t.detach().cpu().numpy()
    else:
        X_np = fit_transform_features_numpy(
            scored[feats].astype(float).values,
            norm,
        )
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
    st.plotly_chart(fig_scatter, width="stretch")
    st.session_state["fig_scatter"] = fig_scatter

    spec_uploaded = st.session_state.get("df_spec")
    cap: pd.DataFrame | None = None
    if spec_uploaded is not None and not spec_uploaded.empty:
        use_auto = st.toggle(
            "Use **auto** limits (mean ± 3σ from data) instead of uploaded spec",
            value=False,
            key="cap_limits_source_uploaded",
        )
        eff_spec = auto_spec_limits(scored, feats) if use_auto else spec_uploaded
        cap = compute_capability(scored, feats, eff_spec)
    else:
        show_auto_cap = st.toggle(
            "Show **Pp / Ppk** using auto limits (mean ± 3σ)",
            value=False,
            key="cap_show_auto_only",
        )
        if show_auto_cap:
            eff_spec = auto_spec_limits(scored, feats)
            cap = compute_capability(scored, feats, eff_spec)

    if cap is not None and not cap.empty:
        st.subheader("Process capability (Pp / Ppk)")
        st.caption(
            "Uses **overall** sample σ — equivalent to Cp/Cpk when no within-subgroup "
            "estimate is available. Ppk >= 1.33 is generally considered capable."
        )
        disp = cap.copy()
        disp["status"] = disp["ppk"].apply(ppk_status)
        st.dataframe(disp, width="stretch")

    st.subheader("Correlation between parameters")
    st.caption(
        "**Spearman (rank) correlation:** strong color = parameters tend to move "
        "together or opposite (order-based; robust to outliers vs linear Pearson). "
        "Blocks show groups that tend to change as a group.",
    )
    fig_corr = make_correlation_heatmap_figure(scored, feats, template=tpl)
    st.plotly_chart(fig_corr, width="stretch")

    st.subheader("Outlier profile (radar)")
    out_idx = scored.index[scored["is_outlier"] == 1].tolist()
    if out_idx and batch_col:
        labels = scored.loc[out_idx, batch_col].astype(str).tolist()
        pick = st.selectbox("Pick an outlier batch", options=labels, key="radar_pick")
        mask = scored[batch_col].astype(str) == pick
        _, z_b = z_scores_for_batch(scored, feats, mask)
        z_b_arr = np.asarray(z_b, dtype=np.float64)
        k_default = min(RADAR_DISPLAY_MAX, len(feats))
        z_map = dict(zip(feats, z_b_arr, strict=True))
        default_sel = sorted(feats, key=lambda f: abs(z_map[f]), reverse=True)[
            :k_default
        ]

        if st.session_state.get("_radar_sync_pick") != pick:
            st.session_state["_radar_sync_pick"] = pick
            st.session_state["radar_metrics_sel"] = default_sel

        st.caption(
            "Defaults highlight the largest deviations vs the population average. "
            f"Add or remove metrics (max {RADAR_DISPLAY_MAX}) for a readable chart—"
            "useful to see one extreme vs several moderate drifts at a glance."
        )
        st.multiselect(
            "Metrics on radar",
            options=sorted(feats),
            key="radar_metrics_sel",
            max_selections=RADAR_DISPLAY_MAX,
            help=(
                "Choose which parameters appear as spokes. Selection resets when you "
                "pick a different outlier batch."
            ),
        )
        selected = list(st.session_state.get("radar_metrics_sel") or [])
        if not selected:
            st.warning(
                "Select at least one metric; showing default deviation highlights.",
            )
            selected = default_sel

        z_sub = subset_z_for_features(feats, z_b_arr, selected)
        z_ref = np.zeros(len(selected))
        alpha_feats = sorted(feats)
        z_alpha = subset_z_for_features(feats, z_b_arr, alpha_feats)
        medians = scored[feats].astype(float).median()
        batch_row = scored.loc[mask, feats].astype(float).mean()
        batch_alpha = np.array([float(batch_row[f]) for f in alpha_feats])
        med_alpha = np.array([float(medians[f]) for f in alpha_feats])
        fig_r = make_radar_figure(
            selected,
            z_ref,
            z_sub,
            template=tpl,
            title=f"Batch {pick} vs average ({len(selected)} metrics)",
        )
        fig_zgrid = make_zscore_grid_figure(
            alpha_feats,
            z_alpha,
            batch_alpha,
            med_alpha,
            template=tpl,
            height=RADAR_FIGURE_HEIGHT,
        )

        col_radar, col_grid = st.columns([3, 2])
        with col_radar:
            st.plotly_chart(fig_r, width="stretch")
        with col_grid:
            st.plotly_chart(fig_zgrid, width="stretch")
            st.caption(
                "All metrics (A–Z): tile color = z-score vs population "
                "(same scale as radar, ±3). Hover for name, batch value, median."
            )
    else:
        st.caption(
            "Run analysis and flag outliers to see a radar chart for a specific batch."
        )

    st.subheader("Distributions per feature")
    hf = st.selectbox("Feature for histogram", options=feats, key="hist_feat")
    fig_h = make_feature_histogram_figure(scored, hf, template=tpl)
    st.plotly_chart(fig_h, width="stretch")

    if dt_col:
        st.subheader("Trend over time")
        res = st.radio(
            "Time grouping", ["day", "month", "hour"], horizontal=True, key="ts_res"
        )
        tf = st.selectbox("Feature for timeline", options=feats, key="tl_feat")

        if res == "hour":
            spec_uploaded = st.session_state.get("df_spec")
            usl_v, lsl_v = _lookup_spec_limits(spec_uploaded, tf)

            xs_tl = baseline_feature_series(
                scored,
                datetime_col=dt_col,
                feature=tf,
                in_control_only=True,
            )
            if len(xs_tl) == 0:
                xs_tl = baseline_feature_series(
                    scored,
                    datetime_col=dt_col,
                    feature=tf,
                    in_control_only=False,
                )
            if len(xs_tl) == 0:
                ucl_v = lcl_v = uwl_v = lwl_v = None
            else:
                m_tl = float(xs_tl.mean())
                sig_tl = float(xs_tl.std(ddof=1)) if len(xs_tl) > 1 else 0.0
                if not (np.isfinite(m_tl) and np.isfinite(sig_tl)):
                    ucl_v = lcl_v = uwl_v = lwl_v = None
                else:
                    ucl_v, lcl_v = m_tl + 3.0 * sig_tl, m_tl - 3.0 * sig_tl
                    uwl_v, lwl_v = m_tl + 2.0 * sig_tl, m_tl - 2.0 * sig_tl

            st.caption(
                "Hourly SPC: **OEG/UCL**, **UEG/LCL** = control limits "
                "(mean +/- 3 sigma, red **thick** dash); **OWG/UWL**, **UWG/LWL** = warning limits "
                "(mean +/- 2 sigma, red **thin** dash); **x̄** = process mean of non-flagged batches "
                "(green solid). **USL/LSL** (blue solid) shown only when a spec-limits file is uploaded."
            )
        else:
            ucl_v = lcl_v = uwl_v = lwl_v = usl_v = lsl_v = None
            st.caption(
                "**x̄** (process mean of non-flagged batches, green) is always shown. "
                "Red control/warning limits and blue USL/LSL appear only for **hour** — "
                "daily/monthly bucket means would not match row-level sigma."
            )

        if not has_in_control_baseline(scored, datetime_col=dt_col, feature=tf):
            st.warning(
                "Every batch is flagged as an outlier — process mean (x̄) is hidden. "
                "Bucket points still show overall movement including flagged batches."
            )

        fig_tl = make_timeline_figure(
            scored,
            datetime_col=dt_col,
            feature=tf,
            batch_col=batch_col,
            resample=res,  # type: ignore[arg-type]
            template=tpl,
            ucl=ucl_v,
            lcl=lcl_v,
            uwl=uwl_v,
            lwl=lwl_v,
            usl=usl_v,
            lsl=lsl_v,
        )
        st.plotly_chart(fig_tl, width="stretch")
        st.session_state["fig_timeline"] = fig_tl

        st.subheader("Production volume")
        st.caption("How many batches per period — normal (grey) vs flagged (orange).")
        fig_prod = make_production_bar_figure(
            scored,
            datetime_col=dt_col,
            resample=res,  # type: ignore[arg-type]
            template=tpl,
        )
        st.plotly_chart(fig_prod, width="stretch")
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
            st.dataframe(comp, width="stretch")
            fig_b = make_delta_bar_figure(comp, template=tpl)
            st.plotly_chart(fig_b, width="stretch")

    st.info(
        "**What LOF means:** each batch gets a score. "
        "Around **1** = similar to neighbors; **well above 1** = unusually isolated — worth a look. "
        "Orange points are above the automatic threshold from your reference data (or from the same file if no reference). "
        "Use the radar and comparison tools to see *which* parameters look off before changing the process.",
    )

    st.divider()
    render_export_section(key_prefix="results_inline_", show_header=True)
