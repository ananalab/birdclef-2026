"""Metriques completes held-out pour les deux pistes (au-dela du ROC-AUC).

Reutilise les probas strictes deja sauvees (_heldout_strict_probs.npy) et
recalcule les probas Perch (LogReg L2), puis calcule pour chaque piste :
  - macro ROC-AUC (rappel, pour coherence),
  - macro Average Precision (mAP) : aire sous precision-rappel, robuste au
    desequilibre (positifs rares),
  - precision / rappel / F1 macro au seuil 0.5,
  - le detail par espece (ROC-AUC + AP).

Memoire-sur : manual lu en colonnes-cles seulement, embeddings focaux mappes
(mmap). Sortie : experiments/metrics.json + _heldout_perch_probs.npy.
"""
import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from src.baselines import predict_sklearn, train_logreg_l2_ovr
from src.config import DATA_PROCESSED, EXPERIMENTS_DIR

DP = DATA_PROCESSED
META_COLS = ("filename", "primary_label", "window_idx", "secondary_labels", "start", "end", "_k")


def feature_cols(df): return [c for c in df.columns if c not in META_COLS]


def build_truth(meta, species):
    idx = {c: i for i, c in enumerate(species)}
    Y = np.zeros((len(meta), len(species)), dtype=np.int32)
    for r, lab in enumerate(meta["primary_label"].astype(str)):
        for c in lab.split(";"):
            if c in idx: Y[r, idx[c]] = 1
    return Y


def heldout_align():
    """Reproduit l'alignement held-out de 09 (739 fenetres uniques)."""
    scm = pd.read_parquet(DP / "features_soundscapes_heldout.parquet")
    spe = np.load(DP / "embeddings_perch_v8_soundscapes.npy")
    spm = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_meta.csv").reset_index(drop=True)
    scm["_k"] = scm["filename"] + "@" + scm["start"].astype(str)
    spm["_k"] = spm["filename"] + "@" + spm["start"].astype(str)
    scm = scm.drop_duplicates("_k").reset_index(drop=True)
    k2r = {}; [k2r.setdefault(k, i) for i, k in enumerate(spm["_k"])]
    common = sorted(set(scm["_k"]) & set(k2r))
    scm = scm.set_index("_k").loc[common].reset_index()
    perch_ho = spe[[k2r[k] for k in common]]
    return scm, perch_ho


def train_perch_focal(species):
    """Entraine LogReg L2 Perch sur tout le focal aligne (manuel+perch)."""
    manual = pd.read_parquet(DP / "features_focal_full_w3_all.parquet",
                             columns=["filename", "window_idx", "primary_label"])
    pm = pd.read_csv(DP / "embeddings_perch_v8_focal_full_meta.csv")
    pm["_r"] = np.arange(len(pm))
    merged = manual.merge(pm[["filename", "window_idx", "_r"]], on=["filename", "window_idx"], how="inner")
    del manual, pm; gc.collect()
    emb = np.load(DP / "embeddings_perch_v8_focal_full.npy", mmap_mode="r")
    X = np.ascontiguousarray(emb[merged["_r"].to_numpy()]).astype(np.float32)
    del emb; gc.collect()
    cols = [f"emb_{i}" for i in range(X.shape[1])]
    models = train_logreg_l2_ovr(pd.DataFrame(X, columns=cols), merged["primary_label"].to_numpy(), species)
    del X; gc.collect()
    return models, cols


def per_class_metrics(Y, P, species, min_pos):
    auc, ap = {}, {}
    for i, c in enumerate(species):
        yt = Y[:, i]; n = int(yt.sum())
        if min_pos <= n < len(yt):
            auc[c] = float(roc_auc_score(yt, P[:, i]))
            ap[c] = float(average_precision_score(yt, P[:, i]))
    return auc, ap


def summarize(Y, P, species, tag):
    auc1, ap1 = per_class_metrics(Y, P, species, 1)
    auc20, ap20 = per_class_metrics(Y, P, species, 20)
    pred = (P >= 0.5).astype(np.int32)
    cols = [i for i, c in enumerate(species) if 1 <= int(Y[:, i].sum()) < len(Y)]
    pr, rc, f1, _ = precision_recall_fscore_support(
        Y[:, cols], pred[:, cols], average="macro", zero_division=0)
    out = {
        "macro_roc_auc_min1": round(float(np.mean(list(auc1.values()))), 4),
        "macro_roc_auc_min20": round(float(np.mean(list(auc20.values()))), 4),
        "macro_ap_min1": round(float(np.mean(list(ap1.values()))), 4),
        "macro_ap_min20": round(float(np.mean(list(ap20.values()))), 4),
        "macro_precision_0.5": round(float(pr), 4),
        "macro_recall_0.5": round(float(rc), 4),
        "macro_f1_0.5": round(float(f1), 4),
        "n_eval_min1": len(auc1), "n_eval_min20": len(auc20),
    }
    per = {c: {"roc_auc": round(auc1[c], 4), "ap": round(ap1[c], 4)} for c in auc1}
    print(f"  {tag}: {out}")
    return out, per


def main():
    t0 = time.time()
    scm, perch_ho = heldout_align()
    print(f"held-out : {len(scm)} fenetres")
    manual_meta = pd.read_parquet(DP / "features_focal_full_w3_all.parquet", columns=["primary_label"])
    species = sorted(set(manual_meta["primary_label"]))
    del manual_meta; gc.collect()
    Y = build_truth(scm, species)

    strict_probs = np.load(DP / "_heldout_strict_probs.npy")
    print("strict probs:", strict_probs.shape)

    perch_path = DP / "_heldout_perch_probs.npy"
    if perch_path.exists():
        perch_probs = np.load(perch_path)
    else:
        models, cols = train_perch_focal(species)
        perch_probs = predict_sklearn(models, pd.DataFrame(perch_ho.astype(np.float32), columns=cols), species)
        np.save(perch_path, perch_probs)
        del models; gc.collect()
    print("perch probs:", perch_probs.shape)

    strict_sum, strict_per = summarize(Y, strict_probs, species, "STRICT")
    perch_sum, perch_per = summarize(Y, perch_probs, species, "PERCH")

    out = {
        "experiment_id": "metrics", "n_heldout_windows": len(scm),
        "scores": {"strict": strict_sum, "perch": perch_sum},
        "per_species": {"strict": strict_per, "perch": perch_per},
    }
    EXP = EXPERIMENTS_DIR / "metrics.json"
    json.dump(out, open(EXP, "w"), indent=2, ensure_ascii=False)
    print(f"Ecrit {EXP} en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
