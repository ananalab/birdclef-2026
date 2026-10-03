import gc, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.config import DATA_PROCESSED, EXPERIMENTS_DIR, RANDOM_SEED

DP = DATA_PROCESSED
LOG = EXPERIMENTS_DIR / "augmented.json"
CAP = 100


def heldout(species):
    scm = pd.read_parquet(DP / "features_soundscapes_heldout.parquet", columns=["filename", "start", "primary_label"])
    spe = np.load(DP / "embeddings_perch_v8_soundscapes.npy").astype(np.float32)
    spm = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_meta.csv").reset_index(drop=True)
    scm["_k"] = scm["filename"] + "@" + scm["start"].astype(str)
    spm["_k"] = spm["filename"] + "@" + spm["start"].astype(str)
    scm = scm.drop_duplicates("_k").reset_index(drop=True)
    k2r = {}; [k2r.setdefault(k, i) for i, k in enumerate(spm["_k"])]
    common = sorted(set(scm["_k"]) & set(k2r))
    scm = scm.set_index("_k").loc[common].reset_index()
    Hy = spe[[k2r[k] for k in common]]
    idx = {c: i for i, c in enumerate(species)}
    Y = np.zeros((len(scm), len(species)), dtype=np.int32)
    for r, lab in enumerate(scm["primary_label"].astype(str)):
        for c in lab.split(";"):
            if c in idx: Y[r, idx[c]] = 1
    return Hy, Y


def cap_mask(meta):
    """Indices a garder : <=CAP fichiers/espece (tries), window_idx<3."""
    meta = meta.reset_index(drop=True)
    keep_files = {}
    rows = []
    for i, r in enumerate(meta.itertuples()):
        if r.window_idx >= 3:
            continue
        fset = keep_files.setdefault(r.primary_label, set())
        if r.filename in fset or len(fset) < CAP:
            fset.add(r.filename); rows.append(i)
    return np.array(rows)


def macro(Y, P, mp):
    vals = []
    for i in range(Y.shape[1]):
        yt = Y[:, i]; n = int(yt.sum())
        if mp <= n < len(yt):
            vals.append(roc_auc_score(yt, P[:, i]))
    return round(float(np.mean(vals)), 4) if vals else float("nan")


def train_predict(X, y, species, Hx_scaled, scaler):
    """LogReg simple par espece sur X deja normalise ; predit sur Hx_scaled."""
    Xs = scaler.transform(X).astype(np.float32)
    P = np.zeros((Hx_scaled.shape[0], len(species)), dtype=np.float32)
    for j, c in enumerate(species):
        yb = (y == c).astype(np.int32)
        if yb.min() == yb.max():
            P[:, j] = float(yb.mean()); continue
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(Xs, yb)
        P[:, j] = clf.predict_proba(Hx_scaled)[:, 1]
    return P


def main():
    t0 = time.time()
    res = json.load(open(LOG)) if LOG.exists() else {"experiment_id": "augmented", "scores": {}}

    # --- augmente ---
    if "perch_aug" not in res["scores"]:
        meta = pd.read_csv(DP / "embeddings_perch_v8_focal_augmented_meta.csv")
        species = sorted(meta["primary_label"].unique())
        Hy, Y = heldout(species)
        rows = cap_mask(meta)
        emb = np.load(DP / "embeddings_perch_v8_focal_augmented.npy", mmap_mode="r")
        X = np.ascontiguousarray(emb[rows]).astype(np.float32); del emb; gc.collect()
        y = meta.iloc[rows]["primary_label"].to_numpy()
        sc = StandardScaler().fit(X)
        Hx = sc.transform(Hy).astype(np.float32)
        P = train_predict(X, y, species, Hx, sc)
        res["scores"]["perch_aug"] = {"macro_min1": macro(Y, P, 1), "macro_min20": macro(Y, P, 20)}
        json.dump(res, open(LOG, "w"), indent=2, ensure_ascii=False)
        print("perch_aug:", res["scores"]["perch_aug"], f"({len(rows)} train, {(time.time()-t0)/60:.1f} min)")
        del X, P, Hx; gc.collect()

    # --- non augmente (memes fenetres <3win, cap identique) ---
    if "perch_noaug_3win" not in res["scores"]:
        meta = pd.read_csv(DP / "embeddings_perch_v8_focal_full_meta.csv")
        species = sorted(meta["primary_label"].unique())
        Hy, Y = heldout(species)
        rows = cap_mask(meta)
        emb = np.load(DP / "embeddings_perch_v8_focal_full.npy", mmap_mode="r")
        X = np.ascontiguousarray(emb[rows]).astype(np.float32); del emb; gc.collect()
        y = meta.iloc[rows]["primary_label"].to_numpy()
        sc = StandardScaler().fit(X)
        Hx = sc.transform(Hy).astype(np.float32)
        P = train_predict(X, y, species, Hx, sc)
        res["scores"]["perch_noaug_3win"] = {"macro_min1": macro(Y, P, 1), "macro_min20": macro(Y, P, 20)}
        json.dump(res, open(LOG, "w"), indent=2, ensure_ascii=False)
        print("perch_noaug_3win:", res["scores"]["perch_noaug_3win"], f"({len(rows)} train, {(time.time()-t0)/60:.1f} min)")

    print("\nScores finaux:", json.dumps(res["scores"], indent=2))


if __name__ == "__main__":
    main()
