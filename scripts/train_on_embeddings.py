import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baselines import predict_sklearn, train_logreg_l2_ovr
from src.config import EXPERIMENTS_DIR
from src.model import predict_proba, train_one_vs_rest
from src.validation import cross_validate_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings", default="data/processed/embeddings_perch_v8_top10.npy")
    p.add_argument("--meta", default="data/processed/embeddings_perch_v8_top10_meta.csv")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=300,
                   help="Nombre d'arbres LightGBM (ignore par LogReg).")
    p.add_argument("--experiment-id", default="perch_models")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print(f"[1/3] Chargement embeddings + meta")
    emb = np.load(args.embeddings)
    meta = pd.read_csv(args.meta)
    print(f"      embeddings shape={emb.shape}, meta shape={meta.shape}")

    # Conversion en DataFrame avec colonnes nommees, comme attendu par
    # cross_validate_model (qui fait X.iloc[idx]).
    X = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    y = meta["primary_label"].to_numpy()
    groups = meta["filename"].to_numpy()
    species = sorted(set(y))
    print(f"      {len(species)} especes, {X.shape[0]} fenetres, {X.shape[1]} features")

    results: dict[str, dict] = {}

    print(f"\n[2/3] LightGBM one-vs-rest (n_estimators={args.n_estimators})")
    t0 = time.time()
    cv_lgbm = cross_validate_model(
        X, y, groups, species,
        n_splits=args.n_splits,
        train_fn=train_one_vs_rest,
        predict_fn=predict_proba,
        n_estimators=args.n_estimators,
        verbose=True,
    )
    cv_lgbm["duration_seconds"] = round(time.time() - t0, 1)
    results["LightGBM"] = cv_lgbm
    print(f"  -> LightGBM : {cv_lgbm['macro_roc_auc_mean']:.4f} "
          f"+/- {cv_lgbm['macro_roc_auc_std']:.4f}")

    print(f"\n[3/3] LogReg L2 (Pipeline StandardScaler + LogisticRegression)")
    t0 = time.time()
    cv_lr = cross_validate_model(
        X, y, groups, species,
        n_splits=args.n_splits,
        train_fn=train_logreg_l2_ovr,
        predict_fn=predict_sklearn,
        n_estimators=0,
        verbose=True,
    )
    cv_lr["duration_seconds"] = round(time.time() - t0, 1)
    results["LogReg L2"] = cv_lr
    print(f"  -> LogReg L2 : {cv_lr['macro_roc_auc_mean']:.4f} "
          f"+/- {cv_lr['macro_roc_auc_std']:.4f}")

    # Tableau comparatif vs references piste stricte
    print("\n--- Comparaison directe avec la piste stricte (features manuelles) ---")
    print(f"  {'Modele':<32} {'Macro AUC CV':>14} {'Std':>8}")
    print(f"  {'-' * 56}")
    print(f"  {'LightGBM tune (manual, S3.2)':<32} {'0.8869':>14} {'0.0236':>8}")
    print(f"  {'LightGBM baseline (manual, S3.1)':<32} {'0.8632':>14} {'0.0174':>8}")
    print(f"  {'LogReg L2 (manual, S3.3)':<32} {'0.8636':>14} {'0.0092':>8}")
    for name, r in results.items():
        print(f"  {name + ' (Perch v8)':<32} "
              f"{r['macro_roc_auc_mean']:>14.4f} {r['macro_roc_auc_std']:>8.4f}")

    duration = time.time() - t_start
    log = {
        "experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "embeddings_path": args.embeddings,
            "meta_path": args.meta,
            "n_splits": args.n_splits,
            "n_estimators_lgbm": args.n_estimators,
            "cv_strategy": "StratifiedGroupKFold groupe par filename, identique sessions 3.1-3.3",
        },
        "dataset": {
            "n_samples": int(X.shape[0]),
            "embedding_dim": int(X.shape[1]),
            "n_species": len(species),
            "species": species,
        },
        "results_by_model": results,
        "reference_manual_features": {
            "lgbm_tuned_cv_mean": 0.8869,
            "lgbm_tuned_cv_std": 0.0236,
            "lgbm_baseline_cv_mean": 0.8632,
            "lgbm_baseline_cv_std": 0.0174,
            "logreg_l2_cv_mean": 0.8636,
            "logreg_l2_cv_std": 0.0092,
        },
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n=== Termine en {duration:.1f}s ===")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
