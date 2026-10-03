import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.baselines import (
    predict_proba_lgbm_np,
    predict_sklearn,
    train_logreg_l2_ovr,
    train_one_vs_rest_parallel,
)
from src.config import EXPERIMENTS_DIR, DATA_PROCESSED
from src.validation import cross_validate_model

META_COLS = ("filename", "primary_label", "window_idx", "secondary_labels")
MANUAL_PARQUET = DATA_PROCESSED / "features_focal_full_w3_all.parquet"
PERCH_NPY = DATA_PROCESSED / "embeddings_perch_v8_focal_full.npy"
PERCH_META = DATA_PROCESSED / "embeddings_perch_v8_focal_full_meta.csv"
OPTUNA_LOG = EXPERIMENTS_DIR / "optuna_lgbm.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-id", default="full_volume_comparison")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--perch-lgbm-estimators", type=int, default=300,
                   help="Arbres LightGBM sur Perch (benchmark, defaut 300).")
    p.add_argument("--with-perch-lgbm", action="store_true",
                   help="Inclure le benchmark secondaire Perch+LightGBM "
                        "(par defaut ignore : redondant avec LogReg L2 et trop "
                        "long, ferait depasser le plafond de 9 h).")
    return p.parse_args()


def load_tuned_params() -> tuple[dict, int]:
    """Recupere les hyperparametres LightGBM tunes en Session 3.2."""
    with open(OPTUNA_LOG) as f:
        data = json.load(f)
    params = dict(data["best_trial"]["best_params"])
    n_estimators = params.pop("n_estimators")
    return params, n_estimators


def build_aligned(manual: pd.DataFrame, perch_emb: np.ndarray,
                  perch_meta: pd.DataFrame) -> tuple:
    """Aligne manual et Perch sur les memes (filename, window_idx)."""
    key_cols = ["filename", "window_idx"]
    perch_meta = perch_meta.copy()
    perch_meta["_perch_row"] = np.arange(len(perch_meta))

    merged = manual.merge(perch_meta[key_cols + ["_perch_row"]], on=key_cols, how="inner")
    merged = merged.sort_index().reset_index(drop=True)

    feature_cols = [c for c in manual.columns if c not in META_COLS]
    X_manual = merged[feature_cols].copy()
    X_perch = pd.DataFrame(
        perch_emb[merged["_perch_row"].to_numpy()],
        columns=[f"emb_{i}" for i in range(perch_emb.shape[1])],
    )
    y = merged["primary_label"].to_numpy()
    groups = merged["filename"].to_numpy()
    return X_manual, X_perch, y, groups


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print("[1/4] Chargement features manuelles + embeddings Perch")
    manual = pd.read_parquet(MANUAL_PARQUET)
    perch_emb = np.load(PERCH_NPY)
    perch_meta = pd.read_csv(PERCH_META)
    print(f"      manual {manual.shape}, perch {perch_emb.shape}")

    X_manual, X_perch, y, groups = build_aligned(manual, perch_emb, perch_meta)
    species = sorted(set(y))
    print(f"      apres alignement : {len(y)} fenetres, {len(species)} especes")

    tuned_params, tuned_n_est = load_tuned_params()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"

    # Reprise : on recharge les resultats deja calcules pour ne pas refaire la
    # piste stricte (5.3 h) si elle est deja sauvegardee.
    results: dict[str, dict] = {}
    if log_path.exists():
        with open(log_path) as f:
            results = json.load(f).get("results", {})
        if results:
            print(f"      Reprise : resultats deja presents = {list(results.keys())}")

    def save_log() -> None:
        """Sauvegarde incrementale : ecrit le JSON apres chaque modele calcule."""
        log = {
            "experiment_id": args.experiment_id,
            "timestamp": timestamp,
            "duration_seconds": round(time.time() - t_start, 1),
            "protocol": f"CV {args.n_splits} folds StratifiedGroupKFold par filename, {len(species)} especes one-vs-rest",
            "dataset": {"n_windows_aligned": int(len(y)), "n_species": len(species)},
            "results": results,
        }
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    def strict_train_fn(X_train, y_train, species_, n_estimators):
        return train_one_vs_rest_parallel(
            X_train, y_train, species_, n_estimators=n_estimators, params=tuned_params,
        )

    if "strict_lgbm_tuned_manual" in results:
        print("[2/3] Piste stricte : deja calculee, reutilisee")
    else:
        print(f"[2/3] Piste stricte : LightGBM tune parallelise, CV {args.n_splits} folds ({len(species)} especes)")
        t0 = time.time()
        results["strict_lgbm_tuned_manual"] = cross_validate_model(
            X_manual, y, groups, species, n_splits=args.n_splits,
            train_fn=strict_train_fn, predict_fn=predict_proba_lgbm_np,
            n_estimators=tuned_n_est, verbose=True,
        )
        save_log()  # checkpoint apres la piste stricte (le livrable)
        print(f"      Strict (manuel)  : {results['strict_lgbm_tuned_manual']['macro_roc_auc_mean']:.4f} "
              f"+/- {results['strict_lgbm_tuned_manual']['macro_roc_auc_std']:.4f} ({(time.time()-t0)/60:.1f} min)")

    if "perch_logreg_l2" in results:
        print("[3/3] Benchmark Perch LogReg L2 : deja calcule, reutilise")
    else:
        print(f"[3/3] Benchmark Perch : LogReg L2, CV {args.n_splits} folds")
        t0 = time.time()
        results["perch_logreg_l2"] = cross_validate_model(
            X_perch, y, groups, species, n_splits=args.n_splits,
            train_fn=train_logreg_l2_ovr, predict_fn=predict_sklearn,
            n_estimators=0, verbose=True,
        )
        save_log()
        print(f"      Perch + LogReg L2 : {results['perch_logreg_l2']['macro_roc_auc_mean']:.4f} "
              f"+/- {results['perch_logreg_l2']['macro_roc_auc_std']:.4f} ({(time.time()-t0)/60:.1f} min)")

    if args.with_perch_lgbm:
        print(f"[bonus] Benchmark Perch : LightGBM ({args.perch_lgbm_estimators} arbres), CV {args.n_splits} folds")
        t0 = time.time()
        results["perch_lightgbm"] = cross_validate_model(
            X_perch, y, groups, species, n_splits=args.n_splits,
            train_fn=train_one_vs_rest_parallel, predict_fn=predict_proba_lgbm_np,
            n_estimators=args.perch_lgbm_estimators, verbose=True,
        )
        save_log()
        print(f"      Perch + LightGBM  : {results['perch_lightgbm']['macro_roc_auc_mean']:.4f} "
              f"+/- {results['perch_lightgbm']['macro_roc_auc_std']:.4f} ({(time.time()-t0)/60:.1f} min)")

    print(f"\n--- Comparaison full volume ({len(species)} especes, CV {args.n_splits} folds) ---")
    print(f"  {'Piste / modele':<32} {'Macro ROC-AUC':>14} {'Std':>8}")
    print(f"  {'-'*56}")
    s = results["strict_lgbm_tuned_manual"]
    print(f"  {'STRICT : LightGBM tune (manuel)':<32} {s['macro_roc_auc_mean']:>14.4f} {s['macro_roc_auc_std']:>8.4f}")
    lr = results["perch_logreg_l2"]
    print(f"  {'BENCH  : LogReg L2 (Perch)':<32} {lr['macro_roc_auc_mean']:>14.4f} {lr['macro_roc_auc_std']:>8.4f}")
    if "perch_lightgbm" in results:
        lg = results["perch_lightgbm"]
        print(f"  {'BENCH  : LightGBM (Perch)':<32} {lg['macro_roc_auc_mean']:>14.4f} {lg['macro_roc_auc_std']:>8.4f}")

    print(f"\n=== Termine en {(time.time()-t_start)/60:.1f} min ===")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
