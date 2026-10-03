"""Extraction Perch v8 sur l'ensemble du focal BirdCLEF 2026.

Dataset de fondation pour le Jour 4 (modele hybride, pseudo-labeling,
ensemble). Pour chaque fichier focal de train.csv :
    - lecture audio 32 kHz mono
    - exclusion si RMS brut < 0.001 (critere EDA des 312 fichiers
      quasi-silencieux)
    - normalisation RMS
    - decoupe en jusqu'a 8 fenetres de 5 s (les premieres ; derniere
      zero-paddee si besoin)
    - 1 embedding Perch 1280D par fenetre

Sortie dans /kaggle/working/ :
    - embeddings_perch_v8_focal_full.npy : (N, 1280) float32
    - embeddings_perch_v8_focal_full_meta.csv : filename, primary_label,
      secondary_labels, window_idx (aligne ligne a ligne avec le npy)

secondary_labels est conserve pour permettre le passage en multi-label
(primary ∪ secondary) cote entrainement, sans re-extraction.

Sauvegarde incrementale tous les CHECKPOINT_EVERY fichiers pour ne rien
perdre si la session Kaggle approche de la limite de 12 h.
"""
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_PATH = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier/8"
OUTPUT_DIR = Path("/kaggle/working")

SAMPLE_RATE = 32_000
WINDOW_DURATION = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)

# Parametres extraction
MAX_WINDOWS = 8           # jusqu'a 8 fenetres par fichier (40 s couvertes)
RMS_MIN = 0.001           # seuil d'exclusion des fichiers quasi-silencieux (EDA)
CHECKPOINT_EVERY = 2000   # sauvegarde intermediaire tous les N fichiers


def raw_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def normalize_rms(audio: np.ndarray, target: float = 0.1) -> np.ndarray:
    rms = raw_rms(audio)
    if rms < 1e-6:
        return audio
    return audio * (target / rms)


def make_windows(audio: np.ndarray, max_windows: int = MAX_WINDOWS) -> list[np.ndarray]:
    """Decoupe en jusqu'a max_windows fenetres 5 s, zero-pad la derniere."""
    windows: list[np.ndarray] = []
    for i in range(max_windows):
        start = i * WINDOW_SAMPLES
        if start >= len(audio):
            break
        end = start + WINDOW_SAMPLES
        if end > len(audio):
            chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            chunk[: len(audio) - start] = audio[start:]
        else:
            chunk = audio[start:end].astype(np.float32)
        windows.append(chunk)
    return windows


def extract_embedding(model, window: np.ndarray) -> np.ndarray:
    win_tensor = tf.constant(window[None, :], dtype=tf.float32)
    output = model.infer_tf(win_tensor)
    if isinstance(output, dict):
        emb = output.get("embedding")
        if emb is None:
            emb = next(v for k, v in output.items() if "embed" in k.lower())
    elif isinstance(output, (tuple, list)):
        emb = output[1] if len(output) >= 2 else output[0]
    else:
        emb = output
    return emb.numpy()[0].astype(np.float32)


def save_outputs(embeddings: list[np.ndarray], meta_rows: list[dict], suffix: str = "") -> None:
    arr = np.array(embeddings, dtype=np.float32)
    np.save(OUTPUT_DIR / f"embeddings_perch_v8_focal_full{suffix}.npy", arr)
    pd.DataFrame(meta_rows).to_csv(
        OUTPUT_DIR / f"embeddings_perch_v8_focal_full{suffix}_meta.csv", index=False)


def main() -> None:
    t_start = time.time()

    for path in (DATA_ROOT, Path(MODEL_PATH)):
        if not path.exists():
            print(f"ERREUR : path introuvable {path}")
            sys.exit(1)

    print("Lecture train.csv")
    df = pd.read_csv(DATA_ROOT / "train.csv")
    cols = ["filename", "primary_label"]
    has_secondary = "secondary_labels" in df.columns
    if has_secondary:
        cols.append("secondary_labels")
    df = df[cols]
    print(f"  {len(df)} fichiers focaux, secondary_labels disponible : {has_secondary}")

    print(f"Chargement Perch v8 depuis {MODEL_PATH}")
    model = hub.load(MODEL_PATH)
    print("Modele charge.")

    audio_root = DATA_ROOT / "train_audio"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    n_errors = 0
    n_excluded_silent = 0
    t_extract = time.time()

    for i, row in enumerate(df.itertuples()):
        if i % 500 == 0:
            elapsed = time.time() - t_extract
            print(f"  {i}/{len(df)} fichiers, {len(all_embeddings)} fenetres, "
                  f"{n_excluded_silent} silencieux exclus, {n_errors} erreurs, "
                  f"{elapsed:.0f}s")

        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save_outputs(all_embeddings, meta_rows)
            print(f"  [checkpoint] sauvegarde a {i} fichiers "
                  f"({len(all_embeddings)} fenetres)")

        audio_path = audio_root / row.filename
        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            if raw_rms(audio) < RMS_MIN:
                n_excluded_silent += 1
                continue
            audio = normalize_rms(audio)
            for w_idx, window in enumerate(make_windows(audio)):
                emb = extract_embedding(model, window)
                all_embeddings.append(emb)
                meta = {
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                    "window_idx": w_idx,
                }
                if has_secondary:
                    meta["secondary_labels"] = row.secondary_labels
                meta_rows.append(meta)
        except Exception as err:
            n_errors += 1
            if n_errors <= 50:
                print(f"  [ERREUR] {row.filename}: {err}")

    extract_duration = time.time() - t_extract
    print(f"\nExtraction terminee :")
    print(f"  {len(all_embeddings)} fenetres extraites")
    print(f"  {n_excluded_silent} fichiers exclus (silencieux)")
    print(f"  {n_errors} erreurs de lecture")
    print(f"  duree {extract_duration:.0f}s "
          f"({extract_duration / max(len(all_embeddings), 1) * 1000:.0f} ms/fenetre)")

    save_outputs(all_embeddings, meta_rows)
    print(f"\nSauvegarde finale OK")
    print(f"\n=== Termine en {(time.time() - t_start):.0f}s ===")


if __name__ == "__main__":
    main()
