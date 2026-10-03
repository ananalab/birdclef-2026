"""Formatage du fichier de soumission au format BirdCLEF+ 2026.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAMPLE_SUBMISSION_CSV, WINDOW_DURATION


def load_submission_template() -> tuple[list[str], list[str]]:
    """Lit la première ligne de sample_submission.csv pour récupérer l'ordre des colonnes.

    On ne lit que l'en-tête (nrows=0) parce que le fichier complet fait
    plusieurs Mo et on n'a besoin que des noms de colonnes.

    Returns:
        (row_id_col, species_cols) : nom de la colonne d'identifiant et liste
        ordonnée des 234 codes espèces tels qu'attendus dans la soumission.
    """
    header = pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=0)
    columns = header.columns.tolist()
    return columns[0], columns[1:]


def build_row_ids(filename_stem: str, n_windows: int) -> list[str]:
    """Génère les row_ids pour les n_windows fenêtres 5 s d'un fichier.

    Convention BirdCLEF : le suffixe est la seconde de FIN de la fenêtre.
    Pour un soundscape de 60 s on aura donc les suffixes 5, 10, 15, ..., 60.

    Args:
        filename_stem: nom du fichier sans extension (ex: BC2026_Train_0001_S08_...).
        n_windows: nombre de fenêtres extraites du fichier.

    Returns:
        Liste de chaînes row_id.
    """
    return [
        f"{filename_stem}_{int((i + 1) * WINDOW_DURATION)}"
        for i in range(n_windows)
    ]


def format_submission(
    predictions_per_file: dict[str, np.ndarray],
    trained_species: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Construit et sauvegarde le submission.csv au format Kaggle.

    Args:
        predictions_per_file: dict {filename_stem: array (n_windows, n_trained_species)}
            des probabilités prédites par le modèle pour chaque fichier soundscape.
        trained_species: liste des codes espèces effectivement entraînées
            (longueur = nb de colonnes dans les arrays de predictions_per_file).
            Doit suivre le même ordre que les colonnes des arrays.
        output_path: chemin du fichier CSV à écrire.

    Returns:
        Le DataFrame écrit, pour inspection.
    """
    row_id_col, all_species = load_submission_template()

    # Construction de l'index de chaque espèce entraînée dans la liste complète.
    # Les espèces entraînées qui ne sont pas dans la taxonomie officielle (cas
    # improbable mais on garde l'assertion) feraient planter ici.
    species_to_col = {code: i for i, code in enumerate(all_species)}
    for code in trained_species:
        assert code in species_to_col, f"Espèce {code} absente du sample_submission.csv"

    rows = []
    row_ids = []
    for filename_stem, probs in predictions_per_file.items():
        n_windows = probs.shape[0]
        ids = build_row_ids(filename_stem, n_windows)
        row_ids.extend(ids)

        # Une ligne par fenêtre, 234 colonnes à 0.0 par défaut. On remplit les
        # 10 colonnes des espèces entraînées avec les probabilités du modèle.
        block = np.zeros((n_windows, len(all_species)), dtype=np.float32)
        for j, code in enumerate(trained_species):
            block[:, species_to_col[code]] = probs[:, j]
        rows.append(block)

    full_matrix = np.vstack(rows)
    df = pd.DataFrame(full_matrix, columns=all_species)
    df.insert(0, row_id_col, row_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
