import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baselines import BENCHMARK_MODELS, predict_sklearn
from src.config import EXPERIMENTS_DIR
from src.validation import cross_validate_model


META_COLS = ("filename", "primary_label", "window_idx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features-path", required=True, type=str,
                   help="Parquet produit par scripts/extract_features.py.")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Nombre de folds CV (defaut: 5, coherent avec sessions 3.1/3.2).")
    p.add_argument("--experiment-id", default="benchmark_models",
                   help="Identifiant pour le log JSON.")
    p.add_argument("--lgbm-tuned-ref", default="experiments/optuna_lgbm.json",
                   help="Chemin du log Session 3.2 pour reprendre les scores LightGBM tune.")
    return p.parse_args()


def split_features_meta(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df[feature_cols].copy(), df["primary_label"].to_numpy(), df["filename"].to_numpy()


def load_lgbm_tuned_reference(path: Path) -> dict:
    """Charge les scores LightGBM tune (Session 3.2) pour comparaison directe."""
    if not path.exists():
        return {"available": False, "reason": f"fichier introuvable : {path}"}
    with open(path) as f:
        data = json.load(f)
    return {
        "available": True,
        "macro_roc_auc_cv_mean": data["best_trial"]["macro_roc_auc_cv_mean"],
        "macro_roc_auc_cv_std": data["best_trial"]["macro_roc_auc_cv_std"],
        "macro_per_fold": data["best_trial"]["macro_per_fold"],
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print(f"[1/3] Chargement {args.features_path}")
    df = pd.read_parquet(args.features_path)
    X, y, groups = split_features_meta(df)
    species = sorted(set(y))
    print(f"      shape={df.shape}, {len(species)} especes, {X.shape[1]} features")

    # On reprend la reference LightGBM tune (pas de re-execution, on garde le
    # resultat de la Session 3.2 pour eviter 2h30 supplementaires).
    lgbm_tuned_ref = load_lgbm_tuned_reference(Path(args.lgbm_tuned_ref))
    if lgbm_tuned_ref["available"]:
        ref_mean = lgbm_tuned_ref["macro_roc_auc_cv_mean"]
        ref_std = lgbm_tuned_ref["macro_roc_auc_cv_std"]
        print(f"      Reference LightGBM tune : {ref_mean:.4f} +/- {ref_std:.4f}")

    # LightGBM baseline (Session 3.1) hard-codee pour eviter une re-execution
    # de la CV 5 folds (deja faite). Reprise du log experiments/cv_lgbm.json.
    lgbm_baseline_ref = {"macro_roc_auc_mean": 0.8632, "macro_roc_auc_std": 0.0174}

    print(f"\n[2/3] Benchmark {len(BENCHMARK_MODELS)} modeles concurrents (CV {args.n_splits} folds)")
    results = {}
    for name, train_fn, n_estimators in BENCHMARK_MODELS:
        print(f"\n  --- {name} (n_estimators={n_estimators if n_estimators else 'n/a'}) ---")
        t_model = time.time()
        cv_results = cross_validate_model(
            X, y, groups, species,
            n_splits=args.n_splits,
            train_fn=train_fn,
            predict_fn=predict_sklearn,
            n_estimators=n_estimators,
            verbose=True,
        )
        duration_model = time.time() - t_model
        cv_results["duration_seconds"] = round(duration_model, 1)
        results[name] = cv_results
        print(
            f"  -> {name} : {cv_results['macro_roc_auc_mean']:.4f} "
            f"+/- {cv_results['macro_roc_auc_std']:.4f} ({duration_model:.1f}s)"
        )

    print("\n[3/3] Tableau comparatif et log JSON")
    print(f"\n  {'Modele':<22} {'Macro AUC CV':>14} {'Std':>8} {'Duree (s)':>10}")
    print(f"  {'-' * 56}")
    print(
        f"  {'LightGBM baseline':<22} "
        f"{lgbm_baseline_ref['macro_roc_auc_mean']:>14.4f} "
        f"{lgbm_baseline_ref['macro_roc_auc_std']:>8.4f} "
        f"{'-':>10}"
    )
    if lgbm_tuned_ref["available"]:
        print(
            f"  {'LightGBM tune':<22} "
            f"{lgbm_tuned_ref['macro_roc_auc_cv_mean']:>14.4f} "
            f"{lgbm_tuned_ref['macro_roc_auc_cv_std']:>8.4f} "
            f"{'-':>10}"
        )
    for name, _, _ in BENCHMARK_MODELS:
        r = results[name]
        print(
            f"  {name:<22} {r['macro_roc_auc_mean']:>14.4f} "
            f"{r['macro_roc_auc_std']:>8.4f} {r['duration_seconds']:>10.1f}"
        )

    duration = time.time() - t_start
    log = {
        "experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "features_path": args.features_path,
            "n_splits": args.n_splits,
            "cv_strategy": "StratifiedGroupKFold groupe par filename, identique sessions 3.1/3.2",
        },
        "dataset": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_species": len(species),
            "species": species,
        },
        "results_by_model": results,
        "reference_lgbm_baseline_cv": lgbm_baseline_ref,
        "reference_lgbm_tuned_cv": lgbm_tuned_ref,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n=== Termine en {duration:.1f}s ({duration / 60:.1f} min) ===")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
