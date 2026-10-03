import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from src.audio import load_audio, normalize_rms
from src.baselines import (
    predict_proba_lgbm_np, predict_sklearn,
    train_logreg_l2_ovr, train_one_vs_rest_parallel,
)
from src.config import DATA_PROCESSED, DATASET_DIR, EXPERIMENTS_DIR, SAMPLE_RATE, WINDOW_SAMPLES
from src.features import extract_features

DP = DATA_PROCESSED
PROB_THRESHOLD = 0.9
CAP_PER_SPECIES = 100
LOG = EXPERIMENTS_DIR / "pseudo_label.json"
PL_CSV = DP / "_pseudo_labels.csv"
PL_STRICT_FEAT = DP / "_pseudo_strict_features.parquet"
SC_DIR = DATASET_DIR / "train_soundscapes"
META_COLS = ("filename", "primary_label", "window_idx", "secondary_labels", "start", "end", "_k")


def feat_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def load_tuned():
    d = json.load(open(EXPERIMENTS_DIR / "optuna_lgbm.json"))
    p = dict(d["best_trial"]["best_params"]); return p, p.pop("n_estimators")


def align_focal():
    manual = pd.read_parquet(DP / "features_focal_full_w3_all.parquet")
    pe = np.load(DP / "embeddings_perch_v8_focal_full.npy")
    pm = pd.read_csv(DP / "embeddings_perch_v8_focal_full_meta.csv"); pm["_r"] = np.arange(len(pm))
    m = manual.merge(pm[["filename", "window_idx", "_r"]], on=["filename", "window_idx"], how="inner")
    fc = feat_cols(manual)
    return m[fc].copy(), pd.DataFrame(pe[m["_r"].to_numpy()], columns=[f"emb_{i}" for i in range(pe.shape[1])]), m["primary_label"].to_numpy(), fc


def heldout():
    scm = pd.read_parquet(DP / "features_soundscapes_heldout.parquet")
    spe = np.load(DP / "embeddings_perch_v8_soundscapes.npy")
    spm = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_meta.csv").reset_index(drop=True)
    scm["_k"] = scm["filename"] + "@" + scm["start"].astype(str)
    spm["_k"] = spm["filename"] + "@" + spm["start"].astype(str)
    scm = scm.drop_duplicates("_k").reset_index(drop=True)
    k2r = {}; [k2r.setdefault(k, i) for i, k in enumerate(spm["_k"])]
    common = sorted(set(scm["_k"]) & set(k2r))
    scm = scm.set_index("_k").loc[common].reset_index()
    return scm, spe[[k2r[k] for k in common]]


def build_truth(meta, species):
    idx = {c: i for i, c in enumerate(species)}
    Y = np.zeros((len(meta), len(species)), dtype=np.int32)
    for r, lab in enumerate(meta["primary_label"].astype(str)):
        for c in lab.split(";"):
            if c in idx:
                Y[r, idx[c]] = 1
    return Y


def macro(Y, probs, species, min_pos):
    per = {}
    for i, c in enumerate(species):
        yt = Y[:, i]; n = int(yt.sum())
        per[c] = float(roc_auc_score(yt, probs[:, i])) if 0 < n < len(yt) and n >= min_pos else float("nan")
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), len([v for v in per.values() if not np.isnan(v)])


def to_sec(hms):
    h, m, s = (int(x) for x in str(hms).split(":")); return h * 3600 + m * 60 + s


def get_window(audio, start_sec):
    start = start_sec * SAMPLE_RATE; end = start + WINDOW_SAMPLES
    if start >= len(audio):
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if end > len(audio):
        c = np.zeros(WINDOW_SAMPLES, dtype=np.float32); c[:len(audio) - start] = audio[start:]; return c
    return audio[start:end].astype(np.float32)


def save_log(d):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(LOG, "w"), indent=2, ensure_ascii=False)


def main():
    t0 = time.time()
    results = json.load(open(LOG)) if LOG.exists() else {"experiment_id": "pseudo_label", "scores": {}}

    X_man, X_perch, y, fc = align_focal()
    species = sorted(set(y))
    scm, sc_perch = heldout()
    Y_ho = build_truth(scm, species)
    X_ho_man = scm[fc].copy()
    sc_perch_df = pd.DataFrame(sc_perch, columns=[f"emb_{i}" for i in range(sc_perch.shape[1])])
    tuned, tuned_n = load_tuned()

    # --- A. Pseudo-labels via Perch LogReg ---
    if PL_CSV.exists():
        print("[A] pseudo-labels deja generes")
        pl = pd.read_csv(PL_CSV)
    else:
        print("[A] Entrainement Perch LogReg sur focal + prediction soundscapes non annotes")
        un_emb = np.load(DP / "embeddings_perch_v8_soundscapes_unlabeled.npy")
        un_meta = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_unlabeled_meta.csv")
        perch_base = train_logreg_l2_ovr(X_perch, y, species)
        un_df = pd.DataFrame(un_emb, columns=[f"emb_{i}" for i in range(un_emb.shape[1])])
        probs = predict_sklearn(perch_base, un_df, species)
        best_i = probs.argmax(1); best_p = probs.max(1)
        sel = pd.DataFrame({"filename": un_meta["filename"], "start": un_meta["start"],
                            "pseudo": [species[i] for i in best_i], "prob": best_p})
        sel = sel[sel["prob"] >= PROB_THRESHOLD]
        sel = sel.groupby("pseudo", group_keys=False).head(CAP_PER_SPECIES).reset_index(drop=True)
        sel.to_csv(PL_CSV, index=False)
        pl = sel
        del un_emb, un_df, perch_base, probs; gc.collect()
    print(f"    {len(pl)} pseudo-labels, {pl['pseudo'].nunique()} especes")
    results["n_pseudo_labels"] = int(len(pl))

    # --- B. Piste Perch : reentrainer focal + pseudo, eval held-out ---
    if "perch_pseudo" not in results["scores"]:
        print("[B] Piste Perch : reentrainement focal + pseudo-labels")
        un_emb = np.load(DP / "embeddings_perch_v8_soundscapes_unlabeled.npy")
        un_meta = pd.read_csv(DP / "embeddings_perch_v8_soundscapes_unlabeled_meta.csv")
        un_meta["_k"] = un_meta["filename"] + "@" + un_meta["start"].astype(str)
        pl["_k"] = pl["filename"] + "@" + pl["start"].astype(str)
        k2r = {}; [k2r.setdefault(k, i) for i, k in enumerate(un_meta["_k"])]
        rows = [k2r[k] for k in pl["_k"] if k in k2r]
        X_aug = pd.concat([X_perch, pd.DataFrame(un_emb[rows], columns=X_perch.columns)], ignore_index=True)
        y_aug = np.concatenate([y, pl["pseudo"].to_numpy()])
        m = train_logreg_l2_ovr(X_aug, y_aug, species)
        p = predict_sklearn(m, sc_perch_df, species)
        m1, n1 = macro(Y_ho, p, species, 1); m20, _ = macro(Y_ho, p, species, 20)
        results["scores"]["perch_pseudo"] = {"macro_min1": round(m1, 4), "macro_min20": round(m20, 4)}
        save_log(results)
        print(f"    Perch + pseudo : held-out {m1:.4f} (>=1), {m20:.4f} (>=20)")
        del un_emb, X_aug, m, p; gc.collect()

    # --- C. Piste stricte : features manuelles des pseudo-fenetres ---
    if not PL_STRICT_FEAT.exists():
        print("[C] Extraction features manuelles des fenetres pseudo-labelisees")
        by_file = {}
        for r in pl.to_dict("records"):
            by_file.setdefault(r["filename"], []).append(r)

        def proc(fn, rows):
            try:
                sig = normalize_rms(load_audio(SC_DIR / fn))
            except Exception:
                return []
            out = []
            for r in rows:
                try:
                    f = extract_features(get_window(sig, int(r["start"])))
                except Exception:
                    continue
                f["primary_label"] = r["pseudo"]; out.append(f)
            return out

        res = Parallel(n_jobs=-1, verbose=5)(delayed(proc)(fn, rows) for fn, rows in by_file.items())
        rows = [x for sub in res for x in sub]
        pd.DataFrame(rows).to_parquet(PL_STRICT_FEAT, index=False)
        print(f"    {len(rows)} fenetres pseudo extraites")

    if "strict_pseudo" not in results["scores"]:
        print("[C] Piste stricte : reentrainement focal + pseudo-labels")
        pf = pd.read_parquet(PL_STRICT_FEAT)
        X_aug = pd.concat([X_man, pf[fc]], ignore_index=True)
        y_aug = np.concatenate([y, pf["primary_label"].to_numpy()])
        m = train_one_vs_rest_parallel(X_aug, y_aug, species, n_estimators=tuned_n, params=tuned)
        p = predict_proba_lgbm_np(m, X_ho_man, species)
        m1, _ = macro(Y_ho, p, species, 1); m20, _ = macro(Y_ho, p, species, 20)
        results["scores"]["strict_pseudo"] = {"macro_min1": round(m1, 4), "macro_min20": round(m20, 4)}
        save_log(results)
        print(f"    Strict + pseudo : held-out {m1:.4f} (>=1), {m20:.4f} (>=20)")

    # Rappel des scores de reference (sans pseudo-label)
    results["reference_heldout_sans_pseudo"] = {
        "strict": {"macro_min1": 0.6104, "macro_min20": 0.5754},
        "perch": {"macro_min1": 0.7460, "macro_min20": 0.7721},
    }
    save_log(results)
    print(f"\n=== Termine en {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
