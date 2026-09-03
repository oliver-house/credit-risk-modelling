import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import (
    COST_FN,
    COST_FP,
    N_FOLDS,
    N_SCORE_BANDS,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COL,
)
from src.evaluation import (
    calibration_slope,
    calibration_table,
    fit_logistic_baseline,
    optimal_threshold,
    score_bands,
    summarise,
    threshold_sensitivity,
)
from src.utils.helpers import get_logger, timer

logger = get_logger(__name__)


def load_scores(data_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    holdout_path = data_dir / "holdout_predictions.csv"
    if holdout_path.exists():
        frame = pd.read_csv(holdout_path)
        return frame[TARGET_COL].to_numpy(), frame["ensemble_holdout"].to_numpy(), "holdout"

    oof_path = data_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise SystemExit(f"No predictions under {data_dir}. Run train.py first.")

    logger.warning(
        f"No {holdout_path.name} — falling back to the out-of-fold predictions. "
        f"Those are the rows the blend weights were tuned on, so every number "
        f"below is optimistic. Re-run train.py with a holdout to fix that."
    )
    frame = pd.read_csv(oof_path)
    return frame[TARGET_COL].to_numpy(), frame["ensemble_oof"].to_numpy(), "oof"


def plot_calibration(tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    _, ax = plt.subplots(figsize=(7, 6))
    for label, table in tables.items():
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o",
                lw=1.5, label=label)

    upper = max(t[["mean_predicted", "observed_rate"]].to_numpy().max()
                for t in tables.values())
    ax.plot([0, upper], [0, upper], linestyle="--", color="grey", lw=1,
            label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by equal-count score bin")
    ax.legend(loc="upper left")
    out_path = out_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Calibration curve saved to {out_path}")
    return out_path


def render_bands(bands: pd.DataFrame) -> str:
    header = ("| Band | n | Score range | Bad rate | Lift | Cumulative bad capture |\n"
              "|------|---|-------------|----------|------|------------------------|")
    rows = [
        f"| {int(r.band)} | {int(r.n):,} | {r.min_score:.4f} to {r.max_score:.4f} | "
        f"{r.bad_rate:.2%} | {r.lift:.2f}x | {r.cum_bad_capture:.1%} |"
        for r in bands.itertuples()
    ]
    return "\n".join([header, *rows])


def _load_feature_cache(data_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    dev = data_dir / "features_dev.npy"
    holdout = data_dir / "features_holdout.npy"
    if dev.exists() and holdout.exists():
        return np.load(dev), np.load(holdout)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Evaluate the smoke run rather than the committed one")
    parser.add_argument("--bands", type=int, default=N_SCORE_BANDS,
                        help=f"Number of equal-sized score bands (default {N_SCORE_BANDS})")
    parser.add_argument("--cost-fn", type=float, default=COST_FN,
                        help=f"Cost of approving a defaulter (default {COST_FN})")
    parser.add_argument("--cost-fp", type=float, default=COST_FP,
                        help=f"Cost of declining a good applicant (default {COST_FP})")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the logistic regression comparison")
    args = parser.parse_args()

    data_dir    = PREDICTIONS_DIR / "smoke" if args.smoke else PREDICTIONS_DIR
    reports_dir = REPORTS_DIR / "smoke"     if args.smoke else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    y, scores, source = load_scores(data_dir)
    logger.info(f"Evaluating {len(y):,} {source} rows, {y.mean():.2%} positive")

    report: dict = {"source": source, **summarise(y, scores, "ensemble")}
    logger.info(f"  AUC {report['auc']:.5f} | Gini {report['gini']:.5f} | "
                f"KS {report['ks']:.4f} | Brier {report['brier']:.5f}")

    raw_table = calibration_table(y, scores, n_bins=args.bands)
    tables = {"Ensemble, as scored": raw_table}

    oof_path = data_dir / "oof_predictions.csv"
    if source == "holdout" and oof_path.exists():
        oof = pd.read_csv(oof_path)
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(oof["ensemble_oof"].to_numpy(), oof[TARGET_COL].to_numpy())
        recalibrated = isotonic.predict(scores)
        tables["Isotonic, fitted on OOF"] = calibration_table(
            y, recalibrated, n_bins=args.bands
        )
        report["recalibrated"] = summarise(y, recalibrated, "isotonic")
        logger.info(f"  After isotonic recalibration fitted on the OOF predictions: "
                    f"Brier {report['recalibrated']['brier']:.5f}, slope "
                    f"{report['recalibrated']['calibration_slope']:.3f}")

    report["calibration"] = raw_table.to_dict(orient="records")
    report["calibration_slope"] = calibration_slope(raw_table)
    calibration_path = plot_calibration(tables, reports_dir)

    bands = score_bands(y, scores, n_bands=args.bands)
    report["bands"] = bands.to_dict(orient="records")
    logger.info(f"  Riskiest band: {bands.iloc[0]['bad_rate']:.2%} bad rate, "
                f"{bands.iloc[0]['lift']:.2f}x lift, "
                f"{bands.iloc[0]['cum_bad_capture']:.1%} of all defaults")

    decision = optimal_threshold(y, scores, cost_fn=args.cost_fn, cost_fp=args.cost_fp)
    report["threshold"] = decision.as_dict()
    logger.info(f"  At {args.cost_fn:g}:{args.cost_fp:g} false-negative to "
                f"false-positive cost, decline at {decision.threshold:.4f}: "
                f"{decision.approval_rate:.1%} approved, "
                f"{decision.bad_rate_approved:.2%} bad among them, "
                f"{decision.bad_capture:.1%} of defaults declined")

    sensitivity = threshold_sensitivity(y, scores, cost_fp=args.cost_fp)
    report["threshold_sensitivity"] = sensitivity.to_dict(orient="records")

    X_dev, X_holdout = (None, None) if args.no_baseline else _load_feature_cache(data_dir)
    if X_dev is not None and source == "holdout":
        oof = pd.read_csv(oof_path)
        with timer("Logistic baseline", logger):
            _, baseline_holdout = fit_logistic_baseline(
                X_dev, oof[TARGET_COL].to_numpy(), X_holdout,
                n_folds=N_FOLDS, random_state=RANDOM_STATE,
            )
        baseline = summarise(y, baseline_holdout, "logistic")
        report["baseline"] = baseline
        report["ensemble_gain_over_baseline"] = report["auc"] - baseline["auc"]
        logger.info(f"  Logistic baseline on the same holdout: AUC "
                    f"{baseline['auc']:.5f}, Gini {baseline['gini']:.5f}")
        logger.info(f"  Ensemble buys {report['ensemble_gain_over_baseline']:+.5f} AUC "
                    f"over a regularised scorecard on the same features")
    elif not args.no_baseline:
        logger.warning(
            "No cached feature matrices, so the logistic baseline is skipped. "
            "Re-run train.py with --save-features to enable the comparison."
        )

    bands_path = reports_dir / "score_bands.md"
    bands_path.write_text(
        f"# Score bands ({source}, {len(y):,} rows, {y.mean():.2%} base rate)\n\n"
        f"Band 1 is the highest-scoring, riskiest slice.\n\n"
        f"{render_bands(bands)}\n\n"
        f"Gini {report['gini']:.4f} · KS {report['ks']:.4f} · "
        f"Brier {report['brier']:.5f}\n",
        encoding="utf-8",
    )
    logger.info(f"Score bands saved to {bands_path}")

    report_path = data_dir / "evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info(f"Evaluation saved to {report_path} (curve at {calibration_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
