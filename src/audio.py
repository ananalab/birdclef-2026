"""Lecture et préparation du signal audio.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, WINDOW_SAMPLES


def load_audio(path: Path | str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Lit un fichier audio en mono float32 au sample rate cible.

    L'EDA a confirmé que le dataset BirdCLEF+ 2026 est déjà en 32 kHz mono OGG,
    donc en pratique on ne devrait jamais avoir à resampler ni à passer en mono.
    On garde quand même les deux garde-fous pour que la fonction reste utilisable
    si on lui passe un fichier d'origine différente (test perso, soundscape
    externe, etc.).

    Args:
        path: chemin vers le fichier audio.
        target_sr: sample rate cible en Hz. Par défaut 32 000.

    Returns:
        Tableau 1D float32 du signal mono au sample rate cible.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # Passage stéréo -> mono par moyenne des canaux si nécessaire.
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Resampling uniquement si le sample rate diffère de la cible.
    if sr != target_sr:
        import librosa  # import local : librosa est lourd à charger.
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32, copy=False)


def normalize_rms(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalise le signal pour qu'il ait un RMS cible.

    Le RMS (root mean square) mesure le niveau sonore moyen du signal. L'EDA a
    montré un ratio max/min de 55 376 entre les fichiers du dataset, ce qui veut
    dire que sans normalisation, un modèle pourrait apprendre des features de
    volume plutôt que de contenu acoustique. On ramène chaque fichier au même
    niveau pour ne comparer que le contenu.

    Le facteur 0.1 par défaut donne un signal confortablement audible sans
    saturer (pas de clipping à 1.0). C'est un choix de convention, pas de
    science.

    Args:
        y: signal audio.
        target_rms: niveau RMS cible.

    Returns:
        Signal normalisé. Si le signal est complètement silencieux (RMS == 0),
        retourne le signal inchangé pour éviter une division par zéro.
    """
    current_rms = float(np.sqrt(np.mean(y ** 2)))
    if current_rms == 0.0:
        return y
    return (y * (target_rms / current_rms)).astype(np.float32, copy=False)


def window_audio(
    y: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    pad_last: bool = True,
) -> list[np.ndarray]:
    """Découpe le signal en fenêtres non recouvrantes de longueur fixe.

    L'évaluation BirdCLEF se fait par fenêtre de 5 secondes, donc on aligne
    notre découpage dessus. Pas de recouvrement à ce stade (on en ajoutera
    peut-être plus tard pour densifier les prédictions à l'inférence).

    Args:
        y: signal audio 1D.
        window_samples: longueur d'une fenêtre en échantillons. Par défaut
            160 000 (= 5 s à 32 kHz).
        pad_last: si True, la dernière fenêtre incomplète est complétée par
            des zéros pour faire window_samples. Si False, elle est ignorée.
            On garde True par défaut pour ne perdre aucune information sur
            les fichiers courts.

    Returns:
        Liste de tableaux 1D, chacun de taille exactement window_samples.
    """
    n_full = len(y) // window_samples
    windows = [y[i * window_samples:(i + 1) * window_samples] for i in range(n_full)]

    remainder = len(y) - n_full * window_samples
    if remainder > 0 and pad_last:
        last = np.zeros(window_samples, dtype=np.float32)
        last[:remainder] = y[n_full * window_samples:]
        windows.append(last)

    # Si le signal est plus court qu'une fenêtre, on retourne une seule fenêtre
    # paddée. Cas typique pour les 2 601 focaux < 5 s identifiés à l'EDA.
    if not windows:
        single = np.zeros(window_samples, dtype=np.float32)
        single[:len(y)] = y
        windows = [single]

    return windows
