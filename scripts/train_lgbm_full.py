"""Entraînement LightGBM one-vs-rest sur les features riches du Jour 2.

Charge un parquet de features produit par scripts/extract_features.py,
entraîne 10 LightGBM binaires (1 par espèce) sur les ~969 features, évalue
via un split 80/20 GroupShuffleSplit par filename, et log les scores.

Usage :
    python scripts/train_lgbm_full.py \\
        --features-path data/processed/features_top10_n50_w3_all.parquet
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR, MODELS_DIR
from src.model import (
    macro_roc_auc,
    predict_proba,
    split_train_val,
    train_one_vs_rest,
)


META_COLS = ("filename", "primary_label", "window_idx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-path", required=True, type=str,
                   help="Chemin du parquet produit par extract_features.py.")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="Nombre d'arbres LightGBM (defaut: 300).")
    p.add_argument("--save-models", action="store_true",
                   help="Sauvegarder les 10 boosters dans models/.")
    p.add_argument("--experiment-id", default="lgbm_full",
                   help="Identifiant pour le log JSON et le dossier modeles.")
    return p.parse_args()


def split_features_meta(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sépare le DataFrame parquet en (X, y, groups).

    X = toutes les colonnes sauf les 3 meta. y = primary_label. groups = filename.
    """
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()
    y = df["primary_label"].to_numpy()
    groups = df["filename"].to_numpy()
    return X, y, groups


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print(f"[1/4] Chargement {args.features_path}")
    df = pd.read_parquet(args.features_path)
    X, y, groups = split_features_meta(df)
    species = sorted(set(y))
    print(f"      shape={df.shape}, {len(species)} especes, {X.shape[1]} features")

    print("[2/4] Split 80/20 GroupShuffleSplit par filename")
    train_idx, val_idx = split_train_val(X, y, groups)
    print(f"      train={len(train_idx)}, val={len(val_idx)}")

    print(f"[3/4] Entrainement {len(species)} LightGBM binaires (n_estimators={args.n_estimators})")
    t_train = time.time()
    models = train_one_vs_rest(
        X.iloc[train_idx],
        y[train_idx],
        species,
        n_estimators=args.n_estimators,
    )
    train_duration = time.time() - t_train
    print(f"      Entrainement termine en {train_duration:.1f}s")

    print("[4/4] Evaluation et log")
    val_probs = predict_proba(models, X.iloc[val_idx], species)
    macro, per_species = macro_roc_auc(y[val_idx], val_probs, species)
    print(f"      Macro ROC-AUC val: {macro:.4f}")
    for code, score in per_species.items():
        score_str = f"{score:.3f}" if not np.isnan(score) else "n/a"
        print(f"        {code}: {score_str}")

    # Sauvegarde des modeles si demandé.
    if args.save_models:
        models_dir = MODELS_DIR / args.experiment_id
        models_dir.mkdir(parents=True, exist_ok=True)
        for code, booster in models.items():
            booster.save_model(str(models_dir / f"{code}.txt"))
        print(f"      Modeles sauves dans {models_dir}")

    # Log JSON
    duration = time.time() - t_start
    log = {
        "experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "features_path": args.features_path,
            "n_estimators": args.n_estimators,
            "n_features": int(X.shape[1]),
            "model": "LightGBM one-vs-rest, params par defaut",
        },
        "dataset": {
            "n_samples_total": int(len(X)),
            "n_samples_train": int(len(train_idx)),
            "n_samples_val": int(len(val_idx)),
            "species": species,
        },
        "scores": {
            "macro_roc_auc_val": round(macro, 4),
            "per_species_roc_auc": {
                code: (round(s, 4) if not np.isnan(s) else None)
                for code, s in per_species.items()
            },
        },
        "models_saved": bool(args.save_models),
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n=== Termine en {duration:.1f}s ===")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
