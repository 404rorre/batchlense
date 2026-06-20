"""CLI: calibrate LOF threshold on reference data, score anomalies, write CSV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from backend.helper_modules.numpy_normalize import Normalization
from backend.lof.pipeline import run_lof_on_dataframe

logger = logging.getLogger(__name__)


def run_lof_pipeline(
    reference_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    n_neighbors: int = 20,
    normalization: Normalization = "zscore",
    threshold_sigma: float = 3.0,
) -> None:
    """
    Load reference (clean) data to set a LOF score threshold, then score input rows.

    Threshold = mean(LOF on reference) + threshold_sigma * std(LOF on reference).
    """
    ref_df = pd.read_csv(reference_path)
    inp_df = pd.read_csv(input_path)
    feature_columns = list(inp_df.columns)
    missing = set(feature_columns) - set(ref_df.columns)
    if missing:
        msg = f"Reference CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)

    out_df, threshold, _ = run_lof_on_dataframe(
        inp_df,
        feature_columns,
        reference_df=ref_df[feature_columns],
        normalization=normalization,
        n_neighbors=n_neighbors,
        threshold_sigma=threshold_sigma,
    )
    logger.info("Calibrated LOF threshold: %s", threshold)

    df_anomalies = out_df.copy()
    # Historical CLI column name used a hyphen.
    if "lof_score" in df_anomalies.columns:
        df_anomalies = df_anomalies.rename(columns={"lof_score": "lof-score"})

    detected = df_anomalies.loc[df_anomalies["is_outlier"] == 1]
    logger.info("Detected %s outliers (rows)", len(detected))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_anomalies.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LOF anomaly detection: calibrate threshold on a reference CSV, "
            "then score an input CSV."
        ),
    )
    p.add_argument(
        "-r",
        "--reference_batches",
        type=Path,
        required=True,
        help="CSV of normal / clean data for threshold calibration",
    )
    p.add_argument(
        "-i",
        "--input_batches",
        type=Path,
        required=True,
        help="CSV to score for outliers",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results.csv"),
        help="Output CSV path (default: results.csv)",
    )
    p.add_argument(
        "-k",
        "--n-neighbors",
        type=int,
        default=20,
        metavar="K",
        help="Number of neighbors for LOF (default: 20)",
    )
    p.add_argument(
        "-n",
        "--normalization",
        choices=["minmax", "zscore", "none"],
        default="zscore",
        help="Feature normalization (default: zscore)",
    )
    p.add_argument(
        "--threshold-sigma",
        type=float,
        default=3.0,
        help=(
            "Sigma for threshold = mean + sigma * std on reference LOF scores "
            "(default: 3)"
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        run_lof_pipeline(
            reference_path=args.reference_batches,
            input_path=args.input_batches,
            output_path=args.output,
            n_neighbors=args.n_neighbors,
            normalization=cast(Normalization, args.normalization),
            threshold_sigma=args.threshold_sigma,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
