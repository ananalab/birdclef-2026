import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import pandas as pd

from src.config import EXPERIMENTS_DIR, MODELS_DIR
from src.model import train_one_vs_rest
from src.tuning import run_optuna_study
from src.validation import cross_validate_model


META_COLS = ("filename", "primary_label", "window_idx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features-path", required=True, type=str,
                   help="Parquet produit par scripts/extract_features.py.")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Nombre de trials Optuna (defaut: 50).")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Nombre de folds CV (defaut: 5).")
    p.add_argument("--experiment-id", default="optuna_lgbm",
                   help="Identifiant pour log JSON et dossier modeles.")
    p.add_argument("--save-best-model", action="store_true", default=True,
                   help="Reentrainer et sauver le meilleur modele (defaut: True).")
    return p.parse_args()


def split_features_meta(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Separe le DataFrame parquet en (X, y, groups)."""
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df[feature_cols].copy(), df["primary_label"].to_numpy(), df["filename"].to_numpy()


def serialize_trials(study: optuna.Study) -> list[dict]:
    """Convertit l'historique des trials en liste de dicts JSON-serialisables."""
    out = []
    for t in study.trials:
        out.append({
            "number": t.number,
            "value": round(t.value, 4) if t.value is not None else None,
            "params": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in t.params.items()},
            "macro_std": round(t.user_attrs.get("macro_std", float("nan")), 4)
                         if not np.isnan(t.user_attrs.get("macro_std", float("nan"))) else None,
            "state": str(t.state).split(".")[-1],
        })
    return out


def compute_param_importance(study: optuna.Study) -> dict[str, float]:
    """Calcule l'importance fANOVA de chaque hyperparametre (capture try / except).

    Optuna estime la part de variance des scores expliquee par chaque parametre.
    Necessite >= 2 trials completes ; protege contre les rares cas degeneres.
    """
    try:
        imp = optuna.importance.get_param_importances(study)
        return {k: round(v, 4) for k, v in imp.items()}
    except Exception as err:
        return {"_error": str(err)}


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().isoformat(timespec="seconds")
    t_start = time.time()

    print(f"[1/4] Chargement {args.features_path}")
    df = pd.read_parquet(args.features_path)
    X, y, groups = split_features_meta(df)
    species = sorted(set(y))
    print(f"      shape={df.shape}, {len(species)} especes, {X.shape[1]} features")

    print(f"\n[2/4] Lancement Optuna ({args.n_trials} trials, CV {args.n_splits} folds)")
    t_tune = time.time()
    study = run_optuna_study(
        X, y, groups, species,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        study_name=args.experiment_id,
        verbose=True,
    )
    tune_duration = time.time() - t_tune
    best_value = study.best_value
    best_params = dict(study.best_params)
    print(f"\n  Meilleur score CV : {best_value:.4f}")
    print(f"  Meilleurs params : {best_params}")
    print(f"  Duree tuning : {tune_duration / 60:.1f} min")

    print("\n[3/4] Reentrainement du meilleur modele sur 100% des donnees")
    t_refit = time.time()
    best_n_estimators = best_params.pop("n_estimators")
    final_models = train_one_vs_rest(
        X, y, species,
        n_estimators=best_n_estimators,
        params=best_params,
    )
    if args.save_best_model:
        models_dir = MODELS_DIR / args.experiment_id
        models_dir.mkdir(parents=True, exist_ok=True)
        for code, booster in final_models.items():
            booster.save_model(str(models_dir / f"{code}.txt"))
        print(f"  Modeles sauves dans {models_dir} (1 booster par espece)")
    print(f"  Duree reentrainement : {time.time() - t_refit:.1f}s")

    print("\n[4/4] Calcul importance des hyperparametres et log JSON")
    importance = compute_param_importance(study)
    print(f"  Importance fANOVA : {importance}")

    duration = time.time() - t_start
    # On remet n_estimators dans best_params pour le log final.
    best_params_full = dict(best_params)
    best_params_full["n_estimators"] = best_n_estimators

    log = {
        "experiment_id": args.experiment_id,
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "duration_minutes": round(duration / 60, 1),
        "config": {
            "features_path": args.features_path,
            "n_trials": args.n_trials,
            "n_splits": args.n_splits,
            "sampler": "TPESampler(n_startup_trials=10)",
            "n_hyperparams_tuned": 7,
            "hyperparams_tuned": [
                "learning_rate", "num_leaves", "min_child_samples",
                "feature_fraction", "bagging_fraction", "reg_lambda", "n_estimators",
            ],
        },
        "dataset": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_species": len(species),
            "species": species,
        },
        "best_trial": {
            "number": int(study.best_trial.number),
            "macro_roc_auc_cv_mean": round(best_value, 4),
            "macro_roc_auc_cv_std": round(
                study.best_trial.user_attrs.get("macro_std", float("nan")), 4,
            ),
            "macro_per_fold": study.best_trial.user_attrs.get("macro_per_fold", []),
            "best_params": best_params_full,
        },
        "reference_baseline_cv": {
            "macro_roc_auc_mean": 0.8632,
            "macro_roc_auc_std": 0.0174,
            "note": "Run Session 3.1 sans tuning (params LightGBM defaut).",
        },
        "param_importance_fanova": importance,
        "all_trials": serialize_trials(study),
        "model_saved": bool(args.save_best_model),
        "model_path": str(MODELS_DIR / args.experiment_id) if args.save_best_model else None,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n=== Termine en {duration / 60:.1f} min ===")
    print(f"Log: {log_path}")
    print(f"Gain vs baseline CV (0.8632) : {best_value - 0.8632:+.4f}")


if __name__ == "__main__":
    main()
