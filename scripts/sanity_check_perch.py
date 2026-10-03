"""Sanity checks sur la pipeline Perch : verifier l'absence de bug
avant de prendre les 0.9655 pour argent comptant.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.baselines import predict_sklearn, train_logreg_l2_ovr
from src.config import RANDOM_SEED
from src.validation import cross_validate_model


def load_data():
    emb = np.load("data/processed/embeddings_perch_v8_top10.npy")
    meta = pd.read_csv("data/processed/embeddings_perch_v8_top10_meta.csv")
    parquet = pd.read_parquet("data/processed/features_top10_n50_w3_all.parquet")
    return emb, meta, parquet


def check_1_extra_rows(meta, parquet):
    print("=" * 70)
    print("CHECK 1 : 4 fenetres supplementaires dans Perch vs parquet")
    print("=" * 70)
    key_perch = meta[["filename", "window_idx"]].apply(tuple, axis=1)
    key_parquet = parquet[["filename", "window_idx"]].apply(tuple, axis=1)
    extra = set(key_perch) - set(key_parquet)
    missing = set(key_parquet) - set(key_perch)
    print(f"  Dans Perch mais pas dans parquet : {len(extra)}")
    print(f"  Dans parquet mais pas dans Perch : {len(missing)}")
    if extra:
        # Recouper avec meta pour voir quelles sont ces fenetres
        for fn, w in sorted(extra)[:10]:
            print(f"    [+] {fn} window={w}")
    if missing:
        for fn, w in sorted(missing)[:10]:
            print(f"    [-] {fn} window={w}")
    print()


def check_2_alignment(emb, meta):
    print("=" * 70)
    print("CHECK 2 : alignement npy <-> meta")
    print("=" * 70)
    print(f"  emb shape : {emb.shape}")
    print(f"  meta shape : {meta.shape}")
    print(f"  Aligne sur la longueur : {len(emb) == len(meta)}")
    # Quelques echantillons aleatoires
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(emb), 5, replace=False)
    print(f"  Echantillons (premiers et derniers de la rangee meta) :")
    for i in sample_idx:
        print(f"    [{i:4d}] {meta.iloc[i]['primary_label']} | {meta.iloc[i]['filename']}"
              f" | win={meta.iloc[i]['window_idx']} | emb[0:3]={emb[i, :3]}"
              f" | emb_norm={np.linalg.norm(emb[i]):.3f}")
    print()


def check_3_cv_grouping(meta):
    print("=" * 70)
    print("CHECK 3 : CV grouping par filename (verifier 0 fuite)")
    print("=" * 70)
    y = meta["primary_label"].to_numpy()
    groups = meta["filename"].to_numpy()
    X_dummy = np.zeros((len(y), 1))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X_dummy, y, groups), start=1):
        train_files = set(groups[train_idx])
        val_files = set(groups[val_idx])
        overlap = train_files & val_files
        print(f"  Fold {fold_i} : {len(train_files)} files train, {len(val_files)} files val,"
              f" overlap = {len(overlap)} files")
        if overlap:
            print(f"    !!! ALERTE FUITE : {list(overlap)[:5]}")
    print()


def check_4_shuffled_labels(emb, meta):
    print("=" * 70)
    print("CHECK 4 : test labels permutes (doit donner ~0.50)")
    print("=" * 70)
    X = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    y = meta["primary_label"].to_numpy()
    groups = meta["filename"].to_numpy()
    species = sorted(set(y))

    # Permutation des labels (en preservant les groupes : on permute par filename
    # pour garder la coherence intra-fichier, sinon le grouping ne tient plus)
    rng = np.random.RandomState(123)
    file_to_label = dict(zip(meta["filename"].unique(),
                             rng.permutation(meta["filename"].unique())))
    # Mappe filename -> nouveau filename -> label correspondant
    # Approche simple : on tire un label au hasard par filename, identique pour
    # toutes ses fenetres
    new_file_to_label = {fn: rng.choice(species)
                         for fn in meta["filename"].unique()}
    y_shuffled = np.array([new_file_to_label[fn] for fn in meta["filename"]])

    print(f"  Distribution labels permutes :")
    for code, n in sorted(pd.Series(y_shuffled).value_counts().items()):
        print(f"    {code}: {n}")

    print(f"  Lancement CV 5 folds LogReg L2 sur labels permutes...")
    cv = cross_validate_model(
        X, y_shuffled, groups, species,
        n_splits=5,
        train_fn=train_logreg_l2_ovr,
        predict_fn=predict_sklearn,
        n_estimators=0,
        verbose=False,
    )
    print(f"  Macro ROC-AUC = {cv['macro_roc_auc_mean']:.4f} +/- {cv['macro_roc_auc_std']:.4f}")
    print(f"  >>> Si proche de 0.50, le pipeline est sain.")
    print(f"  >>> Si > 0.60, il y a une fuite quelque part.")
    print()


def check_5_fold_assignment(meta, parquet):
    print("=" * 70)
    print("CHECK 5 : assignation des folds identique Perch vs parquet ?")
    print("=" * 70)
    print("  Si oui : les modeles sont compares sur les memes train/val")
    print("           donc la difference de score est imputable aux features.")
    print()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Folds parquet
    y_p = parquet["primary_label"].to_numpy()
    g_p = parquet["filename"].to_numpy()
    folds_parquet = {}
    for i, (tr, va) in enumerate(splitter.split(np.zeros((len(y_p), 1)), y_p, g_p)):
        folds_parquet[i] = set(g_p[va])

    # Folds Perch (4 lignes en plus, mais memes filenames)
    y_e = meta["primary_label"].to_numpy()
    g_e = meta["filename"].to_numpy()
    folds_perch = {}
    for i, (tr, va) in enumerate(splitter.split(np.zeros((len(y_e), 1)), y_e, g_e)):
        folds_perch[i] = set(g_e[va])

    for i in range(5):
        common = folds_parquet[i] & folds_perch[i]
        only_p = folds_parquet[i] - folds_perch[i]
        only_e = folds_perch[i] - folds_parquet[i]
        print(f"  Fold {i+1} : {len(common)} filenames communs, "
              f"{len(only_p)} parquet-seul, {len(only_e)} Perch-seul")


def main():
    print(f"\nSanity checks Perch v8 vs features manuelles\n")
    emb, meta, parquet = load_data()
    check_1_extra_rows(meta, parquet)
    check_2_alignment(emb, meta)
    check_3_cv_grouping(meta)
    check_4_shuffled_labels(emb, meta)
    check_5_fold_assignment(meta, parquet)


if __name__ == "__main__":
    main()
