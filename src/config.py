"""Constantes globales du projet.

Centraliser ici toutes les valeurs partagées entre modules (chemins, sample rate,
taille de fenêtre, etc.) évite les valeurs en dur disséminées dans le code.
Modifier ici une fois, propager partout.
"""
from pathlib import Path

# --- Chemins ---
# ROOT = racine du projet (résolu à partir de l'emplacement de ce fichier).
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"
SUBMISSIONS_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "report"

# Dataset Kaggle téléchargé localement (dossier birdclef-2026/, ignoré par git).
DATASET_DIR = ROOT / "birdclef-2026"
TRAIN_AUDIO_DIR = DATASET_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATASET_DIR / "train_soundscapes"
TRAIN_CSV = DATASET_DIR / "train.csv"
TAXONOMY_CSV = DATASET_DIR / "taxonomy.csv"
SOUNDSCAPE_LABELS_CSV = DATASET_DIR / "train_soundscapes_labels.csv"
SAMPLE_SUBMISSION_CSV = DATASET_DIR / "sample_submission.csv"

# --- Traitement audio ---
SAMPLE_RATE = 32_000             # Hz. Convention BirdCLEF, ré-échantillonnage cible.
WINDOW_DURATION = 5.0            # secondes. Granularité d'évaluation imposée par la métrique.
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # = 160 000 échantillons par fenêtre.

# --- Reproductibilité ---
RANDOM_SEED = 42                 # à passer à toute fonction stochastique (split, init modèle, etc.).
