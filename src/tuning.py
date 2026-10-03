"""Tuning bayesien des hyperparametres LightGBM via Optuna.

Optimise 7 hyperparametres LightGBM (learning_rate, num_leaves, min_child_samples,
feature_fraction, bagging_fraction, reg_lambda, n_estimators) en maximisant le
macro ROC-AUC obtenu sur la CV 5 folds StratifiedGroupKFold definie dans
src/validation.py. L'algorithme TPE (Tree-structured Parzen Estimator)
d'Optuna concentre ses essais sur les regions de l'espace prometteuses au
fil des trials, ce qui le rend plus efficace qu'une recherche aleatoire ou
en grille pour un budget identique.

Justification du choix des 7 parametres (consensus litterature + communaute
Kaggle sur LightGBM) :
    - learning_rate, num_leaves, min_child_samples : tres impactants, toujours
      tuner.
    - feature_fraction, bagging_fraction : souvent impactants, tuner si possible.
    - reg_lambda : levier de regularisation L2, garde une voie de regularisation
      explicite.
    - n_estimators : choix du nombre d'arbres, fortement couple au learning_rate.

Parametres explicitement exclus :
    - bagging_freq : impact generalement dans le bruit.
    - reg_alpha : redondant avec reg_lambda en pratique (effet domine).
    - max_depth : couple a num_leaves, on tune ce dernier qui est plus naturel.

Le retrait de ces parametres optimise le ratio nombre de trials / dimension
de l'espace (~7 trials par dimension avec 50 trials, contre ~5.5 avec 9).
"""
from __future__ import annotations

import optuna
from optuna.samplers import TPESampler

from src.config import RANDOM_SEED
from src.model import train_one_vs_rest
from src.validation import cross_validate_model


# Reduit le bruit dans la console pendant les trials. INFO afficherait le
# resume de chaque trial, ce qu'on duplique deja avec notre propre print.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    """Definit l'espace de recherche LightGBM (7 hyperparametres).

    Les echelles log sont utilisees pour les parametres dont l'effet est
    multiplicatif (learning_rate, num_leaves, min_child_samples, reg_lambda),
    afin d'echantillonner uniformement en magnitude plutot qu'en valeur
    absolue. Les bornes choisies couvrent l'amplitude raisonnable pour des
    datasets tabulaires de taille modeste (de l'ordre du millier d'exemples
    par espece).
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 100, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }


def make_train_fn_with_params(lgb_params: dict):
    """Construit une train_fn compatible cross_validate_model avec params injectes.

    cross_validate_model attend une fonction de signature
    (X_train, y_train, species, n_estimators) -> models. On enveloppe
    train_one_vs_rest dans une closure qui injecte les hyperparametres
    proposes par Optuna sans toucher au reste du pipeline.
    """
    def train_fn(X_train, y_train, species, n_estimators):
        return train_one_vs_rest(
            X_train, y_train, species,
            n_estimators=n_estimators, params=lgb_params,
        )
    return train_fn


def lgbm_objective(
    trial: optuna.Trial,
    X,
    y,
    groups,
    species: list[str],
    n_splits: int = 5,
) -> float:
    """Fonction objectif appelee par Optuna pour chaque trial.

    Echantillonne les 7 hyperparametres dans leur espace, lance la CV 5 folds
    StratifiedGroupKFold sur les donnees fournies, retourne le macro ROC-AUC
    moyen sur les folds. Optuna cherche a maximiser cette valeur.

    Args:
        trial: objet Optuna qui propose les valeurs et memorise les resultats.
        X, y, groups, species: donnees et metadonnees fixes pour tous les trials.
        n_splits: nombre de folds CV (5 par defaut, cohesion avec Session 3.1).

    Returns:
        Macro ROC-AUC moyen sur les n_splits folds.
    """
    params = suggest_lgbm_params(trial)
    n_estimators = params.pop("n_estimators")
    train_fn = make_train_fn_with_params(params)

    cv_results = cross_validate_model(
        X, y, groups, species,
        n_splits=n_splits,
        train_fn=train_fn,
        n_estimators=n_estimators,
        verbose=False,
    )
    # On expose le std en tant qu'attribut du trial pour le log final.
    trial.set_user_attr("macro_std", cv_results["macro_roc_auc_std"])
    trial.set_user_attr("macro_per_fold", cv_results["macro_per_fold"])
    return cv_results["macro_roc_auc_mean"]


def run_optuna_study(
    X,
    y,
    groups,
    species: list[str],
    n_trials: int = 50,
    n_splits: int = 5,
    study_name: str = "lgbm_tuning",
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> optuna.Study:
    """Cree et execute l'etude Optuna complete.

    Utilise un sampler TPE seede pour la reproductibilite. Les 10 premiers
    trials sont tires aleatoirement (phase d'exploration "warm-up" du TPE),
    les 40 suivants sont guides par l'historique des resultats.

    Args:
        X, y, groups, species: donnees pour le tuning.
        n_trials: nombre total d'essais (50 par defaut).
        n_splits: folds CV par trial.
        study_name: identifiant de l'etude (utile si stockage SQLite ulterieur).
        seed: graine du sampler TPE pour reproductibilite.
        verbose: affiche un resume par trial si True.

    Returns:
        L'objet optuna.Study complete, qui expose best_trial, trials, etc.
    """
    sampler = TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not verbose:
            return
        best = study.best_value
        std = trial.user_attrs.get("macro_std", float("nan"))
        marker = " <-- new best" if trial.value == best else ""
        print(
            f"  Trial {trial.number + 1}/{n_trials} : "
            f"macro={trial.value:.4f} (std={std:.4f}), best={best:.4f}{marker}"
        )

    study.optimize(
        lambda t: lgbm_objective(t, X, y, groups, species, n_splits=n_splits),
        n_trials=n_trials,
        callbacks=[_callback],
        show_progress_bar=False,
    )
    return study
