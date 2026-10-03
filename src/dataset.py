"""Construction du dataset baseline (X, y, groups) à partir des fichiers audio.

Pour la baseline: on prend les 10 espèces les plus représentées,
50 fichiers par espèce (échantillonnage déterministe), et 1 fenêtre par
fichier (les 5 premières secondes). Pas de multi-label, pas de secondary_labels :
on étiquette uniquement avec primary_label. 
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed

from src.audio import load_audio, normalize_rms, window_audio
from src.config import RANDOM_SEED, TRAIN_AUDIO_DIR, TRAIN_CSV
from src.features import extract_features, extract_mfcc_baseline


def load_train_metadata() -> pd.DataFrame:
    """Charge train.csv et retire les fichiers exclus à l'EDA.

    L'EDA a identifié 312 fichiers avec RMS < 0.001 (quasi-silencieux, probablement
    corrompus) listés dans data/processed/. Pour la baseline on se contente de
    charger le CSV brut et de retirer les colonnes inutiles. On ajoutera la liste
    d'exclusion plus tard si nécessaire.

    Returns:
        DataFrame avec colonnes primary_label, secondary_labels, filename.
    """
    df = pd.read_csv(TRAIN_CSV)
    return df[["primary_label", "secondary_labels", "filename"]].copy()


def select_top_species(df: pd.DataFrame, n: int = 10) -> list[str]:
    """Retourne les n species_code les plus représentés dans train.csv.

    On compte simplement les occurrences de primary_label. Pour la baseline on
    cible les espèces les mieux dotées, où le modèle a une chance d'apprendre
    quelque chose même avec seulement 26 features.

    Args:
        df: train metadata.
        n: nombre d'espèces à retenir.

    Returns:
        Liste des codes espèces triés du plus fréquent au moins fréquent.
    """
    counts = df["primary_label"].value_counts()
    return counts.head(n).index.tolist()


def sample_files_per_species(
    df: pd.DataFrame,
    species: list[str],
    n_files: int = 50,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Échantillonne n_files fichiers par espèce, de manière déterministe.

    Si une espèce a moins de n_files exemples, on prend tout. Le tirage est
    seedé pour que deux exécutions donnent le même sous-ensemble (reproductibilité
    indispensable pour comparer des runs).

    Args:
        df: train metadata complet.
        species: liste des codes espèces à inclure.
        n_files: nombre cible de fichiers par espèce.
        seed: graine du tirage aléatoire.

    Returns:
        Sous-DataFrame contenant au plus n_files * len(species) lignes.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for code in species:
        subset = df[df["primary_label"] == code]
        take = min(n_files, len(subset))
        chosen_idx = rng.choice(subset.index.values, size=take, replace=False)
        parts.append(subset.loc[chosen_idx])
    return pd.concat(parts, ignore_index=True)


def build_baseline_dataset(
    df_sampled: pd.DataFrame,
    audio_root: Path = TRAIN_AUDIO_DIR,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extrait les features de la première fenêtre 5 s de chaque fichier.

    Pour chaque fichier de df_sampled, on lit l'audio, on normalise en RMS, on
    prend la première fenêtre 5 s (paddée si besoin) et on extrait les 26 MFCC
    de baseline. Les fichiers illisibles sont signalés et exclus (pas d'arrêt).

    Args:
        df_sampled: sous-ensemble de train metadata produit par sample_files_per_species.
        audio_root: dossier racine contenant les fichiers audio.

    Returns:
        X: DataFrame (n_samples, 26) des features, indexé séquentiellement.
        y: array (n_samples,) des labels primary_label en string.
        groups: array (n_samples,) des filenames pour le GroupShuffleSplit.
    """
    rows, labels, groups = [], [], []

    for _, meta in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Extraction"):
        audio_path = audio_root / meta["filename"]
        try:
            signal = normalize_rms(load_audio(audio_path))
            first_window = window_audio(signal)[0]
            feats = extract_mfcc_baseline(first_window)
        except Exception as err:
            # Fichier corrompu / illisible : on log et on saute. Pour le baseline
            # on accepte une légère perte. Sur le pipeline complet du Jour 2 on
            # tiendra une vraie liste d'erreurs.
            print(f"[skip] {meta['filename']}: {err}")
            continue

        rows.append(feats)
        labels.append(meta["primary_label"])
        groups.append(meta["filename"])

    X = pd.DataFrame(rows)
    y = np.array(labels)
    groups_arr = np.array(groups)
    return X, y, groups_arr


# --- Extraction riche (Session 2.5) : N fenêtres par fichier, ~969 features ---

def extract_file_windows(
    meta_row: dict,
    audio_root: Path = TRAIN_AUDIO_DIR,
    n_windows: int = 3,
    families: list[str] | None = None,
) -> list[dict] | None:
    """Extrait jusqu'à n_windows fenêtres 5 s d'un fichier et leurs features.

    Cette fonction est appelée en parallèle par build_dataset_rich. Elle est
    autonome (pas d'état partagé) pour être joblib-friendly.

    Args:
        meta_row: dict-like avec clés 'primary_label' et 'filename'.
        audio_root: dossier racine des fichiers audio.
        n_windows: nombre max de fenêtres à extraire (les premières du fichier).
        families: liste de familles de features à calculer. None = toutes (969).

    Returns:
        Liste de dicts (1 par fenêtre) avec features + meta (filename, label,
        window_idx). Retourne None si le fichier est illisible.
    """
    audio_path = audio_root / meta_row["filename"]
    try:
        signal = normalize_rms(load_audio(audio_path))
        windows = window_audio(signal)[:n_windows]
    except Exception:
        return None

    rows = []
    for i, w in enumerate(windows):
        try:
            feats = extract_features(w, families=families)
        except Exception:
            continue
        feats["filename"] = meta_row["filename"]
        feats["primary_label"] = meta_row["primary_label"]
        feats["window_idx"] = i
        rows.append(feats)
    return rows


def build_dataset_rich(
    df_sampled: pd.DataFrame,
    n_windows: int = 3,
    families: list[str] | None = None,
    n_jobs: int = -1,
    audio_root: Path = TRAIN_AUDIO_DIR,
) -> pd.DataFrame:
    """Construit le DataFrame complet (features riches) en parallélisant par fichier.

    Pour chaque fichier de df_sampled, on extrait jusqu'à n_windows fenêtres 5 s
    et leurs features. Le résultat est concaténé en un seul DataFrame avec les
    colonnes meta filename / primary_label / window_idx à la fin.

    Args:
        df_sampled: sous-ensemble de train metadata (sortie de sample_files_per_species).
        n_windows: nombre de fenêtres par fichier.
        families: familles de features (None = les 6 = ~969 features).
        n_jobs: nombre de processus parallèles. -1 = tous les cœurs.
        audio_root: dossier racine des fichiers audio.

    Returns:
        DataFrame (n_samples, n_features + 3 colonnes meta).
    """
    meta_rows = df_sampled.to_dict(orient="records")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_file_windows)(row, audio_root, n_windows, families)
        for row in meta_rows
    )

    all_rows = []
    for r in results:
        if r is not None:
            all_rows.extend(r)

    return pd.DataFrame(all_rows)
