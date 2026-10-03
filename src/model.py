import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import RANDOM_SEED


# Hyperparamètres LightGBM par défaut. Volontairement non tunés pour la baseline.
# Au Jour 3 on utilisera Optuna pour les optimiser proprement.
LGB_BASELINE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}


def split_train_val(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sépare train et val avec GroupShuffleSplit pour éviter la fuite.

    Le grouping par filename est CRUCIAL : deux fenêtres issues du même fichier
    contiennent du contenu acoustique très similaire (souvent le même oiseau qui
    chante). Si on en met une en train et l'autre en val, le modèle a "déjà vu"
    ce contenu et le score val sera artificiellement élevé.

    Pour la baseline on a 1 fenêtre par fichier donc le risque est nul, mais on
    garde le grouping par cohérence avec les sessions suivantes (Jour 2+ on
    aura plusieurs fenêtres par fichier).

    Returns:
        train_idx, val_idx : indices entiers dans X / y.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, val_idx


def train_one_vs_rest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    species: list[str],
    n_estimators: int = 200,
    params: dict | None = None,
) -> dict[str, lgb.Booster]:
    """Entraîne un LightGBM binaire par espèce.

    Pour chaque espèce code S de la liste, on construit un label binaire
    (1 si y_train == S, 0 sinon) et on entraîne un modèle dessus. Le paramètre
    scale_pos_weight équilibre l'effet du déséquilibre fort entre la classe
    positive (les fichiers de S) et la classe négative (tout le reste).

    Args:
        X_train: features d'entraînement.
        y_train: labels primary_label.
        species: liste ordonnée des codes espèces (l'ordre détermine celui
            des colonnes en sortie de predict_proba).
        n_estimators: nombre d'arbres LightGBM. 200 est un compromis vitesse /
            performance raisonnable sans tuning.
        params: dict d'hyperparametres LightGBM. Si None, utilise
            LGB_BASELINE_PARAMS. Les cles "objective", "metric", "verbosity"
            et "seed" sont reinjectees au cas ou le caller les omet, pour
            garantir un comportement coherent (entrainement binaire silencieux
            avec seed projet).

    Returns:
        Dictionnaire {code_espece: booster_lightgbm}.
    """
    models: dict[str, lgb.Booster] = {}
    base_params = dict(params) if params is not None else dict(LGB_BASELINE_PARAMS)
    # Garde-fous : on impose les cles non negociables meme si le caller les omet
    base_params.setdefault("objective", "binary")
    base_params.setdefault("metric", "auc")
    base_params.setdefault("verbosity", -1)
    base_params.setdefault("seed", RANDOM_SEED)

    for code in species:
        y_bin = (y_train == code).astype(np.int32)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)

        # scale_pos_weight = n_neg / n_pos contrebalance le déséquilibre. Si la
        # classe positive est rare, ses erreurs comptent plus dans la loss.
        run_params = dict(base_params)
        run_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        train_data = lgb.Dataset(X_train, label=y_bin)
        booster = lgb.train(run_params, train_data, num_boost_round=n_estimators)
        models[code] = booster

    return models


def predict_proba(
    models: dict[str, lgb.Booster],
    X: pd.DataFrame,
    species: list[str],
) -> np.ndarray:
    """Retourne la matrice de probabilités (n_samples, n_species).

    L'ordre des colonnes suit l'ordre de la liste species, ce qui garantit
    l'alignement avec les opérations en aval (format_submission, calcul de
    score).
    """
    preds = np.zeros((len(X), len(species)), dtype=np.float32)
    for i, code in enumerate(species):
        preds[:, i] = models[code].predict(X)
    return preds


def macro_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    species: list[str],
) -> tuple[float, dict[str, float]]:
    """Calcule le macro-averaged ROC-AUC, comme la métrique officielle BirdCLEF.

    On calcule le ROC-AUC par classe indépendamment puis on moyenne. Les classes
    qui n'ont aucun positif dans y_true sont ignorées (sklearn lève une erreur,
    on attrape avec NaN puis np.nanmean).

    Returns:
        (score_macro, dict_par_espece) : moyenne et détail par classe.
    """
    per_species: dict[str, float] = {}
    for i, code in enumerate(species):
        y_bin = (y_true == code).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_species[code] = float("nan")
            continue
        per_species[code] = float(roc_auc_score(y_bin, y_proba[:, i]))

    macro = float(np.nanmean(list(per_species.values())))
    return macro, per_species
