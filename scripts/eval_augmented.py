import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.baselines import predict_proba_lgbm_np, predict_sklearn, train_logreg_l2_ovr, train_one_vs_rest_parallel
from src.config import DATA_PROCESSED, EXPERIMENTS_DIR

DP = DATA_PROCESSED
LOG = EXPERIMENTS_DIR / "augmented.json"
META = ("filename", "primary_label", "window_idx", "secondary_labels", "start", "end", "_k")


def fcols(df): return [c for c in df.columns if c not in META]
def load_tuned():
    d = json.load(open(EXPERIMENTS_DIR / "optuna_lgbm.json"))
    p = dict(d["best_trial"]["best_params"]); return p, p.pop("n_estimators")


def heldout(species):
    scm = pd.read_parquet(DP / "features_soundscapes_heldout.parquet")
    spe = np.load(DP / "embeddings_perch_v8_soundscapes.npy")
    spm = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_meta.csv").reset_index(drop=True)
    scm["_k"] = scm["filename"] + "@" + scm["start"].astype(str)
    spm["_k"] = spm["filename"] + "@" + spm["start"].astype(str)
    scm = scm.drop_duplicates("_k").reset_index(drop=True)
    k2r = {}; [k2r.setdefault(k, i) for i, k in enumerate(spm["_k"])]
    common = sorted(set(scm["_k"]) & set(k2r))
    scm = scm.set_index("_k").loc[common].reset_index()
    perch = spe[[k2r[k] for k in common]]
    idx = {c: i for i, c in enumerate(species)}
    Y = np.zeros((len(scm), len(species)), dtype=np.int32)
    for r, lab in enumerate(scm["primary_label"].astype(str)):
        for c in lab.split(";"):
            if c in idx: Y[r, idx[c]] = 1
    return scm[fcols(scm)].copy(), perch, Y


def macro(Y, probs, species, mp):
    vals = []
    for i in range(len(species)):
        yt = Y[:, i]; n = int(yt.sum())
        if mp <= n < len(yt):
            vals.append(roc_auc_score(yt, probs[:, i]))
    return float(np.mean(vals)) if vals else float("nan")


def save(d): json.dump(d, open(LOG, "w"), indent=2, ensure_ascii=False)


def main():
    t0 = time.time()
    res = json.load(open(LOG)) if LOG.exists() else {"experiment_id": "augmented", "scores": {}}
    tuned, tn = load_tuned()

    # --- STRICT augmente ---
    if "strict_aug" not in res["scores"]:
        aug = pd.read_parquet(DP / "features_focal_augmented.parquet")
        species = sorted(aug["primary_label"].unique())
        fc = fcols(aug)
        Xho_man, _, Y = heldout(species)
        print(f"[strict] {aug.shape} aug, held-out {len(Y)} fenetres, {len(species)} especes")
        m = train_one_vs_rest_parallel(aug[fc], aug["primary_label"].to_numpy(), species, n_estimators=tn, params=tuned)
        p = predict_proba_lgbm_np(m, Xho_man, species)
        res["scores"]["strict_aug"] = {"macro_min1": round(macro(Y, p, species, 1), 4),
                                       "macro_min20": round(macro(Y, p, species, 20), 4)}
        save(res); del m, aug, p; gc.collect()
        print("  strict_aug:", res["scores"]["strict_aug"])

    # --- PERCH augmente (bloc memoire-sur, mmap + liberation) ---
    if "perch_aug" not in res["scores"]:
        meta_a = pd.read_csv(DP / "embeddings_perch_v8_focal_augmented_meta.csv")
        species = sorted(meta_a["primary_label"].unique())
        Xho_man, perch_ho, Y = heldout(species)
        emb_a = np.load(DP / "embeddings_perch_v8_focal_augmented.npy").astype(np.float32)
        cols = [f"emb_{i}" for i in range(emb_a.shape[1])]
        ho_df = pd.DataFrame(perch_ho.astype(np.float32), columns=cols)
        ma = train_logreg_l2_ovr(pd.DataFrame(emb_a, columns=cols), meta_a["primary_label"].to_numpy(), species)
        del emb_a; gc.collect()
        pa = predict_sklearn(ma, ho_df, species)
        res["scores"]["perch_aug"] = {"macro_min1": round(macro(Y, pa, species, 1), 4),
                                      "macro_min20": round(macro(Y, pa, species, 20), 4)}
        save(res); del ma, pa, ho_df; gc.collect()
        print("  perch_aug:", res["scores"]["perch_aug"])

    # --- PERCH non augmente sur memes fenetres (<3win), bloc separe ---
    if "perch_noaug_3win" not in res["scores"]:
        meta_f = pd.read_csv(DP / "embeddings_perch_v8_focal_full_meta.csv")
        species = sorted(meta_f["primary_label"].unique())
        Xho_man, perch_ho, Y = heldout(species)
        keep = (meta_f["window_idx"] < 3).to_numpy()
        cols = [f"emb_{i}" for i in range(perch_ho.shape[1])]
        ho_df = pd.DataFrame(perch_ho.astype(np.float32), columns=cols)
        # lecture mappee : on ne materialise que le sous-ensemble <3win
        emb_f = np.load(DP / "embeddings_perch_v8_focal_full.npy", mmap_mode="r")
        Xsub = np.ascontiguousarray(emb_f[keep]).astype(np.float32)
        del emb_f; gc.collect()
        mb = train_logreg_l2_ovr(pd.DataFrame(Xsub, columns=cols), meta_f[keep]["primary_label"].to_numpy(), species)
        del Xsub; gc.collect()
        pb = predict_sklearn(mb, ho_df, species)
        res["scores"]["perch_noaug_3win"] = {"macro_min1": round(macro(Y, pb, species, 1), 4),
                                             "macro_min20": round(macro(Y, pb, species, 20), 4)}
        save(res); del mb, pb, ho_df; gc.collect()
        print("  perch_noaug_3win:", res["scores"]["perch_noaug_3win"])

    res["reference_heldout_sans_aug"] = {"strict": {"macro_min1": 0.6104, "macro_min20": 0.5754},
                                         "perch_full8win": {"macro_min1": 0.7460, "macro_min20": 0.7721}}
    save(res)
    print(f"\n=== Termine en {(time.time()-t0)/60:.1f} min ===")
    print(json.dumps(res["scores"], indent=2))


if __name__ == "__main__":
    main()
