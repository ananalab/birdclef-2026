import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR
from src.model import (
    macro_roc_auc,
    predict_proba,
    split_train_val,
    train_one_vs_rest,
)
from src.validation import cross_validate_model


META_COLS = ("filename", "primary_label", "window_idx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features-path", required=True, type=str,
                   help="Chemin du parquet produit par extract_features.py.")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="Nombre d'arbres LightGBM (defaut: 300).")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Nombre de folds CV (defaut: 5).")
    p.add_argument("--experiment-id", default="cv_lgbm",
                   help="Identifiant pour le log JSON.")
    return p.parse_args()


def split_features_meta(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Separe le DataFrame parquet en (X, y, groups)."""
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()
    y = df["primary_label"].to_numpy()
    groups = df["filename"].to_numpy()
    return X, y, groups


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print(f"[1/3] Chargement {args.features_path}")
    df = pd.read_parquet(args.features_path)
    X, y, groups = split_features_meta(df)
    species = sorted(set(y))
    print(f"      shape={df.shape}, {len(species)} especes, {X.shape[1]} features")

    print(f"\n[2/3] Cross-validation StratifiedGroupKFold ({args.n_splits} folds)")
    t_cv = time.time()
    cv_results = cross_validate_model(
        X, y, groups, species,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        verbose=True,
    )
    cv_duration = time.time() - t_cv
    mean_str = f"{cv_results['macro_roc_auc_mean']:.4f}"
    std_str = f"{cv_results['macro_roc_auc_std']:.4f}"
    print(f"\n  Macro ROC-AUC CV : {mean_str} +/- {std_str}")
    print(f"  Duree CV : {cv_duration:.1f}s")

    print("\n[3/3] Reference : split unique 80/20 (pour comparaison avec Jour 2)")
    t_ref = time.time()
    train_idx, val_idx = split_train_val(X, y, groups)
    models = train_one_vs_rest(
        X.iloc[train_idx], y[train_idx], species,
        n_estimators=args.n_estimators,
    )
    val_probs = predict_proba(models, X.iloc[val_idx], species)
    single_macro, single_per_species = macro_roc_auc(y[val_idx], val_probs, species)
    print(f"  Split unique : macro_auc={single_macro:.4f}")
    print(f"  Reference Jour 2 : 0.8981")
    print(f"  Duree ref : {time.time() - t_ref:.1f}s")

    # Log structure JSON
    duration = time.time() - t_start
    log = {
        "experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "features_path": args.features_path,
            "n_estimators": args.n_estimators,
            "n_splits": args.n_splits,
            "model": "LightGBM one-vs-rest, params par defaut",
            "cv_strategy": "StratifiedGroupKFold groupe par filename",
        },
        "dataset": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_species": len(species),
            "species": species,
        },
        "cv_results": cv_results,
        "single_split_reference": {
            "macro_roc_auc": round(single_macro, 4),
            "per_species": {
                c: (round(s, 4) if not np.isnan(s) else None)
                for c, s in single_per_species.items()
            },
            "note": "Comparaison directe avec experiments/lgbm_full.json (0.8981)",
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
