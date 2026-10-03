"""Validation croisee StratifiedGroupKFold pour l'evaluation des modeles.

Module central regroupant le protocole de validation utilise pour comparer
tous les modeles du projet (LightGBM, XGBoost, LogReg, etc.). La strategie
repose sur une StratifiedGroupKFold a 5 folds avec grouping par filename.

Justification du grouping (anti-fuite) : nos fenetres 5 s sont issues du
decoupage de fichiers audio plus longs. Plusieurs fenetres extraites d'un
meme fichier focal partagent un contenu acoustique fortement correle (meme
prise, meme bruit de fond, meme individu chanteur). Toute evaluation non
groupee place certaines de ces fenetres simultanement en train et en val,
gonflant artificiellement les scores. Mesure au Jour 2 sanity check :
+0.04 de macro ROC-AUC sans grouping versus avec.

Justification de la stratification : avec un dataset deja restreint
(50 fichiers par espece) et un fort desequilibre de classes, un GroupKFold
non stratifie peut produire des folds ou une espece rare est totalement
absente de la val, donnant un ROC-AUC NaN sur ce fold pour cette espece
et destabilisant la moyenne macro. La stratification cherche a egaliser
la distribution des classes entre folds tout en respectant la contrainte
de grouping.

Choix de design : les modeles entraines a chaque fold ne sont PAS conserves
sur disque. La CV vise a estimer la performance attendue, pas a produire
un modele utilisable. Le modele final sera reentraine sur 100 pour cent
des donnees apres selection des hyperparametres.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_SEED
from src.model import (
    macro_roc_auc,
    predict_proba,
    train_one_vs_rest,
)


def make_cv_splitter(
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> StratifiedGroupKFold:
    """Construit le splitter StratifiedGroupKFold standard du projet.

    Centraliser la creation du splitter ici garantit que tous les modeles
    compares (LightGBM, XGBoost, LogReg, ensembles) utilisent strictement
    le meme decoupage : condition necessaire pour que les ecarts observes
    soient attribuables aux modeles et non au hasard du split.

    Args:
        n_splits: nombre de folds. 5 est le compromis standard entre
            stabilite de l'estimation et cout calcul (n entrainements).
        seed: graine random pour la reproductibilite du shuffle interne.

    Returns:
        Un objet StratifiedGroupKFold pret a etre itere via .split(X, y, groups).
    """
    return StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed,
    )


def compute_fold_score(
    X: pd.DataFrame,
    y: np.ndarray,
    species: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_fn=train_one_vs_rest,
    predict_fn=predict_proba,
    n_estimators: int = 300,
) -> tuple[float, dict[str, float], dict]:
    """Entraine et evalue un fold unique de la CV.

    Cette fonction est extraite de la boucle CV pour pouvoir etre reutilisee
    avec d'autres familles de modeles (XGBoost, RandomForest, LogReg) au
    Jour 3 : il suffit d'injecter une autre train_fn respectant la signature
    (X_train, y_train, species, n_estimators) -> models et eventuellement
    une predict_fn adaptee si l'API du modele differe de LightGBM.

    Args:
        X: features completes (n_samples, n_features).
        y: labels primary_label (n_samples,).
        species: liste ordonnee des codes especes.
        train_idx: indices du fold d'entrainement.
        val_idx: indices du fold de validation.
        train_fn: fonction d'entrainement, par defaut train_one_vs_rest.
        predict_fn: fonction de prediction probabiliste, par defaut
            src.model.predict_proba (utilise Booster.predict). Pour les
            modeles sklearn / XGBClassifier, fournir une fonction adaptee.
            Doit avoir la signature (models, X, species) -> np.ndarray
            de shape (n_samples, n_species).
        n_estimators: nombre d'arbres (ignore par certains modeles).

    Returns:
        macro_score: macro ROC-AUC sur le fold de val.
        per_species_score: dict {code: roc_auc} avec NaN pour les especes
            sans positif dans le fold.
        info: dict avec n_train, n_val, et la liste des especes degenerees.
    """
    models = train_fn(
        X.iloc[train_idx], y[train_idx], species, n_estimators=n_estimators,
    )
    val_probs = predict_fn(models, X.iloc[val_idx], species)
    macro, per_species = macro_roc_auc(y[val_idx], val_probs, species)

    nan_species = [c for c, s in per_species.items() if np.isnan(s)]
    info = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_species_nan": len(nan_species),
        "species_nan": nan_species,
    }
    return macro, per_species, info


def cross_validate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    species: list[str],
    n_splits: int = 5,
    train_fn=train_one_vs_rest,
    predict_fn=predict_proba,
    n_estimators: int = 300,
    verbose: bool = True,
) -> dict:
    """Execute la CV complete et agrege les scores en un dict serialisable.

    Pour chaque fold du splitter : entraine le modele sur le sous-ensemble
    train, predit sur le sous-ensemble val, calcule le macro ROC-AUC. Les
    modeles ne sont pas conserves (cf. choix de design en tete de module).

    Le std est calcule avec ddof=1 (estimateur sans biais de l'ecart-type
    d'un echantillon), convention standard pour des resultats experimentaux
    rapportes dans un rapport.

    Args:
        X: features (n_samples, n_features).
        y: labels primary_label.
        groups: array des filenames, base du grouping anti-fuite.
        species: codes especes ordonnes.
        n_splits: nombre de folds.
        train_fn: fonction d'entrainement injectable.
        predict_fn: fonction de prediction probabiliste injectable. Permet de
            supporter les modeles a API differente (sklearn predict_proba
            renvoie (n, 2) versus lightgbm Booster.predict qui renvoie (n,)).
        n_estimators: hyperparametre forward au train_fn.
        verbose: affiche un resume par fold si True.

    Returns:
        Dict structure pour log JSON direct :
            - macro_roc_auc_mean, macro_roc_auc_std
            - macro_per_fold: liste des 5 scores
            - per_species_mean, per_species_std: moyennes par espece
              (calculees en ignorant les NaN eventuels)
            - fold_info: liste des info par fold (tailles, especes NaN)
    """
    splitter = make_cv_splitter(n_splits=n_splits)

    fold_macros: list[float] = []
    fold_per_species: list[dict[str, float]] = []
    fold_infos: list[dict] = []

    iterator = splitter.split(X, y, groups=groups)
    for fold_i, (train_idx, val_idx) in enumerate(iterator, start=1):
        macro, per_species, info = compute_fold_score(
            X, y, species, train_idx, val_idx,
            train_fn=train_fn, predict_fn=predict_fn, n_estimators=n_estimators,
        )
        fold_macros.append(macro)
        fold_per_species.append(per_species)
        fold_infos.append(info)

        if verbose:
            nan_msg = f", {info['n_species_nan']} esp NaN" if info["n_species_nan"] else ""
            print(
                f"  Fold {fold_i}/{n_splits} : train={info['n_train']}, "
                f"val={info['n_val']}, macro_auc={macro:.4f}{nan_msg}"
            )

    macro_mean = float(np.mean(fold_macros))
    macro_std = float(np.std(fold_macros, ddof=1)) if len(fold_macros) > 1 else 0.0

    per_species_mean: dict[str, float] = {}
    per_species_std: dict[str, float] = {}
    for code in species:
        scores = [fold[code] for fold in fold_per_species]
        valid = [s for s in scores if not np.isnan(s)]
        if valid:
            per_species_mean[code] = float(np.mean(valid))
            per_species_std[code] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        else:
            per_species_mean[code] = float("nan")
            per_species_std[code] = float("nan")

    return {
        "n_splits": n_splits,
        "macro_roc_auc_mean": round(macro_mean, 4),
        "macro_roc_auc_std": round(macro_std, 4),
        "macro_per_fold": [round(m, 4) for m in fold_macros],
        "per_species_mean": {
            c: (round(v, 4) if not np.isnan(v) else None)
            for c, v in per_species_mean.items()
        },
        "per_species_std": {
            c: (round(v, 4) if not np.isnan(v) else None)
            for c, v in per_species_std.items()
        },
        "fold_info": fold_infos,
    }
