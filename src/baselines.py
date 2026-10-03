"""Modeles classiques comparatifs : XGBoost, RandomForest, LogisticRegression.

Toutes les fonctions de ce module respectent strictement la signature
attendue par src.validation.cross_validate_model :

    train_fn(X_train, y_train, species, n_estimators) -> dict[str, classifier]

Et leurs predictions sont recuperees via predict_sklearn (defini ici), qui
fait l'adaptation de l'API sklearn (predict_proba renvoie (n, 2) avec la
proba de la classe positive en colonne 1) vers le format attendu par
src.validation (un array (n_samples, n_species)).

Choix de design:
    - Tous les modeles sont en strategie one-vs-rest sans optimisation
      multilabel, pour rester aligne avec le pipeline LightGBM existant
      et permettre les comparaisons directes au point de vue protocolaire.
    - Les hyperparametres sont fixes a des valeurs raisonnables documentees
      (pas de tuning sur ces concurrents). LightGBM est le challenger principal,
      ces baselines servent de "tour d'horizon" pour le rapport, pas de
      candidats au modele final. Tuner chacun individuellement consommerait
      des heures de calcul pour un gain attendu marginal compte tenu de la
      hierarchie generale entre familles de modeles tabulaires
      (gradient boosting > random forest > logreg en regle generale sur
      features audio manuelles).
    - La LogReg est encapsulee dans un Pipeline avec StandardScaler car les
      methodes lineaires sont tres sensibles a l'echelle des features. Les
      MFCC et indices bioacoustiques ont des magnitudes tres differentes
      (RMS dans [0,1], MFCC en dizaines, bandwidth en milliers) : sans
      scaling, la LogReg serait dominee par les features de grande amplitude.
      Les modeles arborescents (LightGBM, XGBoost, RF) n'ont pas ce probleme
      car leurs splits sont invariants par changement d'echelle monotone.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.model import LGB_BASELINE_PARAMS


# -- XGBoost ----------------------------------------------------------------

# Hyperparametres XGBoost raisonnables. Volontairement proches des defauts
# LightGBM utilises en baseline (learning_rate, max_depth) pour limiter
# l'effet "tuning implicite par mes choix manuels".
XGB_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": "hist",   # algorithme rapide, equivalent au "hist" de LightGBM
    "verbosity": 0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def train_xgb_ovr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 300,
) -> dict[str, xgb.XGBClassifier]:
    """Entraine un XGBClassifier binaire par espece (one-vs-rest).

    Strictement parallele a train_one_vs_rest de src.model mais avec XGBoost
    a la place de LightGBM. Le scale_pos_weight contrebalance le desequilibre
    classe positive / negative (10 espece contre 9 autres).
    """
    models: dict[str, xgb.XGBClassifier] = {}
    for code in species:
        y_bin = (y_train == code).astype(np.int32)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)

        params = dict(XGB_DEFAULT_PARAMS)
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        params["n_estimators"] = n_estimators

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_bin)
        models[code] = clf
    return models


# -- RandomForest -----------------------------------------------------------

RF_DEFAULT_PARAMS = {
    "max_depth": None,            # arbres complets, regularisation par min_samples_leaf
    "min_samples_leaf": 2,
    "max_features": "sqrt",       # standard pour la classification
    "class_weight": "balanced",   # equivalent au scale_pos_weight des boosters
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


def train_rf_ovr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 300,
) -> dict[str, RandomForestClassifier]:
    """Entraine un RandomForestClassifier binaire par espece.

    Note : sklearn supporte nativement le multiclasse, mais on garde la
    boucle one-vs-rest pour rester compatible avec le pipeline et les
    metriques du projet (et pouvoir potentiellement reentrainer une espece
    individuelle a part).
    """
    models: dict[str, RandomForestClassifier] = {}
    for code in species:
        y_bin = (y_train == code).astype(np.int32)
        clf = RandomForestClassifier(
            n_estimators=n_estimators, **RF_DEFAULT_PARAMS,
        )
        clf.fit(X_train, y_bin)
        models[code] = clf
    return models


# -- LogReg L1 et L2 --------------------------------------------------------

LOGREG_COMMON = {
    "max_iter": 5000,             # marge pour convergence sur 969 features
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}


class _ConstantProbaModel:
    """Predit une probabilite constante. Utilise quand une espece n'a aucun
    (ou que des) positif dans le fold d'entrainement : LogisticRegression
    exige au moins 2 classes, contrairement a LightGBM. On retombe alors sur
    la frequence de base de la classe, ce qui donne un score non informatif
    (AUC ~0.5) pour cette espece sur ce fold, sans planter le pipeline.
    """

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X) -> np.ndarray:
        out = np.zeros((len(X), 2), dtype=np.float32)
        out[:, 1] = self.p
        out[:, 0] = 1.0 - self.p
        return out


def _build_logreg_pipeline(penalty: str) -> Pipeline:
    """Construit le Pipeline scaler + LogisticRegression pour une penalite donnee.

    Le solver depend de la penalite : liblinear pour L1 (un des seuls
    a la supporter), lbfgs pour L2 (plus rapide sur ce cas).
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(penalty=penalty, solver=solver, **LOGREG_COMMON)),
    ])


def _fit_logreg_ovr(X_train, y_train, species, penalty: str) -> dict:
    """Boucle one-vs-rest commune L1/L2, robuste aux especes mono-classe."""
    models: dict = {}
    for code in species:
        y_bin = (y_train == code).astype(np.int32)
        if y_bin.min() == y_bin.max():
            # Une seule classe presente dans ce fold : modele constant.
            models[code] = _ConstantProbaModel(float(y_bin.mean()))
            continue
        pipe = _build_logreg_pipeline(penalty)
        pipe.fit(X_train, y_bin)
        models[code] = pipe
    return models


def train_logreg_l1_ovr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 0,  # ignore
) -> dict:
    """Entraine une LogReg avec penalite L1 (selection de features implicite)."""
    return _fit_logreg_ovr(X_train, y_train, species, "l1")


def train_logreg_l2_ovr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 0,  # ignore
) -> dict:
    """Entraine une LogReg avec penalite L2 (regularisation par retrecissement)."""
    return _fit_logreg_ovr(X_train, y_train, species, "l2")


# -- LightGBM one-vs-rest parallelise (par espece) --------------------------

# train_one_vs_rest de src.model entraine les especes sequentiellement, chaque
# LightGBM utilisant son multithreading interne. Sur des problemes binaires de
# taille moderee (dizaines de milliers de lignes), ce multithreading ne sature
# pas les coeurs (~1,3 coeur effectif observe). Sur 206 especes x 5 folds, c'est
# trop lent. Ici on inverse : chaque LightGBM en mono-thread, et on parallelise
# l'entrainement des especes sur tous les coeurs via joblib. Speedup ~6x.

def _fit_binary_booster(X_np: np.ndarray, y_bin: np.ndarray,
                        n_estimators: int, params: dict) -> lgb.Booster:
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)
    run = dict(params)
    run["num_threads"] = 1
    run["scale_pos_weight"] = n_neg / max(n_pos, 1)
    run.setdefault("objective", "binary")
    run.setdefault("metric", "auc")
    run.setdefault("verbosity", -1)
    run.setdefault("seed", RANDOM_SEED)
    return lgb.train(run, lgb.Dataset(X_np, label=y_bin), num_boost_round=n_estimators)


def train_one_vs_rest_parallel(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 300,
    params: dict | None = None,
    n_jobs: int = -1,
) -> dict[str, lgb.Booster]:
    """LightGBM one-vs-rest entraine en parallele sur les especes.

    X est converti une fois en float32 numpy ; joblib (loky) le memory-mappe
    donc il est partage entre workers sans copie. Chaque worker entraine un
    LightGBM mono-thread pour une espece. La prediction doit ensuite passer
    par predict_proba_lgbm_np (entrainement sur numpy => predire sur numpy,
    pour eviter le mismatch de noms de colonnes).
    """
    base = dict(params) if params is not None else dict(LGB_BASELINE_PARAMS)
    X_np = X_train.to_numpy(dtype=np.float32) if hasattr(X_train, "to_numpy") else np.asarray(X_train, dtype=np.float32)
    y_bins = [(y_train == code).astype(np.int32) for code in species]
    boosters = Parallel(n_jobs=n_jobs)(
        delayed(_fit_binary_booster)(X_np, yb, n_estimators, base) for yb in y_bins
    )
    return dict(zip(species, boosters))


def predict_proba_lgbm_np(models: dict, X: pd.DataFrame, species: list[str]) -> np.ndarray:
    """predict_proba pour boosters entraines sur numpy (cf. train_one_vs_rest_parallel)."""
    X_np = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
    preds = np.zeros((len(X), len(species)), dtype=np.float32)
    for i, code in enumerate(species):
        preds[:, i] = models[code].predict(X_np)
    return preds


# -- Predict adapter --------------------------------------------------------

def predict_sklearn(models: dict, X: pd.DataFrame, species: list[str]) -> np.ndarray:
    """Recupere les probabilites de classe positive pour chaque modele sklearn.

    Tous les estimateurs sklearn et XGBClassifier renvoient via predict_proba
    une matrice (n_samples, 2) : la colonne 0 contient P(classe negative),
    la colonne 1 contient P(classe positive). On extrait la colonne 1 pour
    chaque espece et on concatene au format attendu par macro_roc_auc.

    L'ordre des colonnes en sortie suit l'ordre de la liste species, comme
    pour src.model.predict_proba.
    """
    preds = np.zeros((len(X), len(species)), dtype=np.float32)
    for i, code in enumerate(species):
        preds[:, i] = models[code].predict_proba(X)[:, 1]
    return preds


# -- Registry des modeles a benchmarker -------------------------------------

# Centralise les couples (train_fn, n_estimators_default) pour le script
# de benchmark. n_estimators=0 pour les modeles qui ne l'utilisent pas
# permet d'eviter la confusion ("c'est ignore, pas un vrai 0").
BENCHMARK_MODELS = [
    ("XGBoost",           train_xgb_ovr,        300),
    ("RandomForest",      train_rf_ovr,         300),
    ("LogReg L1",         train_logreg_l1_ovr,  0),
    ("LogReg L2",         train_logreg_l2_ovr,  0),
]
