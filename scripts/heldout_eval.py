import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.baselines import predict_proba_lgbm_np, predict_sklearn, train_logreg_l2_ovr, train_one_vs_rest_parallel
from src.config import DATA_PROCESSED, EXPERIMENTS_DIR, MODELS_DIR

FOCAL_MANUAL = DATA_PROCESSED / "features_focal_full_w3_all.parquet"
FOCAL_PERCH_NPY = DATA_PROCESSED / "embeddings_perch_v8_focal_full.npy"
FOCAL_PERCH_META = DATA_PROCESSED / "embeddings_perch_v8_focal_full_meta.csv"
SC_MANUAL = DATA_PROCESSED / "features_soundscapes_heldout.parquet"
SC_PERCH_NPY = DATA_PROCESSED / "embeddings_perch_v8_soundscapes.npy"
SC_PERCH_META = DATA_PROCESSED / "embeddings_perch_v8_soundscapes_meta.csv"
OPTUNA_LOG = EXPERIMENTS_DIR / "optuna_lgbm.json"
META_COLS = ("filename", "primary_label", "window_idx", "secondary_labels", "start", "end", "_k")


def load_tuned_params():
    d = json.load(open(OPTUNA_LOG))
    p = dict(d["best_trial"]["best_params"])
    return p, p.pop("n_estimators")


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def align_focal(manual, perch_emb, perch_meta):
    """Aligne le focal manuel et Perch sur les memes (filename, window_idx)."""
    pm = perch_meta.copy()
    pm["_r"] = np.arange(len(pm))
    merged = manual.merge(pm[["filename", "window_idx", "_r"]], on=["filename", "window_idx"], how="inner")
    fc = feature_cols(manual)
    X_manual = merged[fc].copy()
    X_perch = pd.DataFrame(perch_emb[merged["_r"].to_numpy()],
                           columns=[f"emb_{i}" for i in range(perch_emb.shape[1])])
    y = merged["primary_label"].to_numpy()
    return X_manual, X_perch, y


def build_truth(meta_sc, species):
    """Matrice multi-label (n_fenetres, n_especes) depuis primary_label ';'."""
    idx = {c: i for i, c in enumerate(species)}
    Y = np.zeros((len(meta_sc), len(species)), dtype=np.int32)
    for r, lab in enumerate(meta_sc["primary_label"].astype(str)):
        for code in lab.split(";"):
            if code in idx:
                Y[r, idx[code]] = 1
    return Y


def macro_multilabel(Y_true, probs, species, min_pos=1):
    per = {}
    for i, code in enumerate(species):
        yt = Y_true[:, i]
        npos = int(yt.sum())
        if npos < min_pos or npos == len(yt):
            per[code] = float("nan")
        else:
            per[code] = float(roc_auc_score(yt, probs[:, i]))
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per, len(vals)


def main():
    t_start = time.time()
    timestamp = datetime.now().isoformat(timespec="seconds")

    print("[1/5] Chargement focal (train) et soundscape (held-out)")
    manual = pd.read_parquet(FOCAL_MANUAL)
    perch_emb = np.load(FOCAL_PERCH_NPY)
    perch_meta = pd.read_csv(FOCAL_PERCH_META)
    X_manual, X_perch, y = align_focal(manual, perch_emb, perch_meta)
    species = sorted(set(y))
    print(f"      train aligne : {len(y)} fenetres, {len(species)} especes")
    # Liberer les gros tableaux focaux sources (manual ~350 Mo, perch_emb ~890 Mo)
    del manual, perch_emb, perch_meta
    gc.collect()

    sc_manual = pd.read_parquet(SC_MANUAL)
    sc_perch_emb = np.load(SC_PERCH_NPY)
    sc_perch_meta = pd.read_csv(SC_PERCH_META).reset_index(drop=True)
    # Le CSV d'annotations liste chaque fenetre 2x a l'identique (739 uniques
    # x 2 = 1478). On deduplique sur (filename, start) pour ne pas double-compter.
    sc_manual["_k"] = sc_manual["filename"] + "@" + sc_manual["start"].astype(str)
    sc_perch_meta["_k"] = sc_perch_meta["filename"] + "@" + sc_perch_meta["start"].astype(str)
    sc_manual = sc_manual.drop_duplicates("_k").reset_index(drop=True)
    perch_k_to_row = {}
    for i, k in enumerate(sc_perch_meta["_k"]):
        perch_k_to_row.setdefault(k, i)  # premiere occurrence
    common = sorted(set(sc_manual["_k"]) & set(perch_k_to_row))
    sc_manual = sc_manual.set_index("_k").loc[common].reset_index()
    sc_perch_aligned = sc_perch_emb[[perch_k_to_row[k] for k in common]]
    print(f"      held-out aligne : {len(common)} fenetres uniques (deduplique de {len(sc_perch_emb)})")

    # Verite terrain (depuis la meta manuelle, identique a Perch)
    Y_true = build_truth(sc_manual, species)
    X_sc_manual = sc_manual[feature_cols(sc_manual)].copy()

    tuned_params, tuned_n_est = load_tuned_params()
    strict_probs_path = DATA_PROCESSED / "_heldout_strict_probs.npy"
    strict_models_dir = MODELS_DIR / "strict_heldout"

    # --- Piste stricte : entrainer, SAUVER les modeles, predire, SAUVER les
    # probas, liberer. Double filet : si les probas existent on saute tout ;
    # sinon si les modeles sont sur disque on saute l'entrainement (88 min) ;
    # sinon on entraine et on sauve les modeles AVANT de predire (pour ne plus
    # rien reperdre sur un bug post-entrainement). Econome en memoire : les
    # 206 boosters (~1.4 Go) sont liberes avant l'entrainement Perch (l'OOM
    # precedent venait de leur cohabitation avec Perch sur 8 Go).
    if strict_probs_path.exists():
        print("[2/5] STRICT : probas held-out deja sauvees, reutilisees")
        strict_probs = np.load(strict_probs_path)
    else:
        saved = sorted(strict_models_dir.glob("*.txt")) if strict_models_dir.exists() else []
        if len(saved) == len(species):
            print(f"[2/5] STRICT : {len(saved)} modeles charges du disque (entrainement saute)")
            strict_models = {p.stem: lgb.Booster(model_file=str(p)) for p in saved}
        else:
            print("[2/5] Entrainement STRICT (LightGBM tune) sur tout le focal")
            t0 = time.time()
            strict_models = train_one_vs_rest_parallel(X_manual, y, species, n_estimators=tuned_n_est, params=tuned_params)
            strict_models_dir.mkdir(parents=True, exist_ok=True)
            for code, booster in strict_models.items():
                booster.save_model(str(strict_models_dir / f"{code}.txt"))
            print(f"      entraine en {(time.time()-t0)/60:.1f} min, modeles sauves")
        strict_probs = predict_proba_lgbm_np(strict_models, X_sc_manual, species)
        np.save(strict_probs_path, strict_probs)
        del strict_models
        gc.collect()

    # Liberer la memoire focale manuelle avant Perch
    del X_manual
    gc.collect()

    print("[3/5] Entrainement BENCH (LogReg L2 Perch) sur tout le focal")
    t0 = time.time()
    perch_models = train_logreg_l2_ovr(X_perch, y, species)
    print(f"      {(time.time()-t0)/60:.1f} min")

    print("[4/5] Prediction Perch sur held-out + scores")
    # Memes noms de colonnes qu'a l'entrainement (emb_0..emb_N) pour eviter le
    # mismatch de feature names de sklearn.
    sc_perch_df = pd.DataFrame(
        sc_perch_aligned, columns=[f"emb_{i}" for i in range(sc_perch_aligned.shape[1])]
    )
    perch_probs = predict_sklearn(perch_models, sc_perch_df, species)

    strict_macro, strict_per, n_eval = macro_multilabel(Y_true, strict_probs, species, min_pos=1)
    perch_macro, perch_per, _ = macro_multilabel(Y_true, perch_probs, species, min_pos=1)
    strict_macro20, _, n_eval20 = macro_multilabel(Y_true, strict_probs, species, min_pos=20)
    perch_macro20, _, _ = macro_multilabel(Y_true, perch_probs, species, min_pos=20)

    print(f"\n--- Held-out soundscape ({len(common)} fenetres) ---")
    print(f"  {'Piste':<34} {'macro (>=1 pos)':>16} {'macro (>=20 pos)':>18}")
    print(f"  {'-'*70}")
    print(f"  {'STRICT : LightGBM tune (manuel)':<34} {strict_macro:>16.4f} {strict_macro20:>18.4f}")
    print(f"  {'BENCH  : LogReg L2 (Perch)':<34} {perch_macro:>16.4f} {perch_macro20:>18.4f}")
    print(f"  especes evaluables : {n_eval} (>=1 pos), {n_eval20} (>=20 pos)")

    log = {
        "experiment_id": "heldout_eval",
        "timestamp": timestamp,
        "duration_seconds": round(time.time() - t_start, 1),
        "n_heldout_windows": len(common),
        "n_species_eval_min1": n_eval,
        "n_species_eval_min20": n_eval20,
        "scores": {
            "strict_lgbm_manual": {"macro_min1": round(strict_macro, 4), "macro_min20": round(strict_macro20, 4)},
            "perch_logreg_l2": {"macro_min1": round(perch_macro, 4), "macro_min20": round(perch_macro20, 4)},
        },
        "per_species": {
            "strict": {c: (round(v, 4) if not np.isnan(v) else None) for c, v in strict_per.items()},
            "perch": {c: (round(v, 4) if not np.isnan(v) else None) for c, v in perch_per.items()},
        },
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_DIR / "heldout_eval.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n=== Termine en {(time.time()-t_start)/60:.1f} min ===")


if __name__ == "__main__":
    main()
