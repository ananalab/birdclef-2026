"""Extraction Perch v8 sur les soundscapes NON annotes (pour pseudo-labeling).

Parcourt tous les soundscapes de train_soundscapes/ sauf les 66 annotes,
extrait l'embedding Perch 1280D de chaque fenetre 5 s. Servira au Jour 4 :
predire avec le meilleur modele, garder les predictions tres confiantes
(>=0.9) comme pseudo-labels, reentrainer dans le domaine cible.

Checkpointing tous les CHECKPOINT_EVERY fichiers (run potentiellement long).
Sortie /kaggle/working/ : embeddings_perch_v8_soundscapes_unlabeled.npy +
_meta.csv (filename, start, end).
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
WINDOW_SAMPLES = int(SAMPLE_RATE * 5.0)
CHECKPOINT_EVERY = 1000


def normalize_rms(a, target=0.1):
    rms = float(np.sqrt(np.mean(a ** 2)))
    return a if rms < 1e-6 else a * (target / rms)


def extract_embedding(model, window):
    out = model.infer_tf(tf.constant(window[None, :], dtype=tf.float32))
    if isinstance(out, dict):
        emb = out.get("embedding")
        if emb is None:
            emb = next(v for k, v in out.items() if "embed" in k.lower())
    elif isinstance(out, (tuple, list)):
        emb = out[1] if len(out) >= 2 else out[0]
    else:
        emb = out
    return emb.numpy()[0].astype(np.float32)


def save(embeddings, meta):
    np.save(OUTPUT_DIR / "embeddings_perch_v8_soundscapes_unlabeled.npy",
            np.array(embeddings, dtype=np.float32))
    pd.DataFrame(meta).to_csv(
        OUTPUT_DIR / "embeddings_perch_v8_soundscapes_unlabeled_meta.csv", index=False)


def main():
    t0 = time.time()
    annotated = set(pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")["filename"].unique())
    sc_dir = DATA_ROOT / "train_soundscapes"
    files = sorted(p.name for p in sc_dir.glob("*.ogg") if p.name not in annotated)
    print(f"{len(files)} soundscapes non annotes a traiter")

    model = hub.load(MODEL_PATH)
    print("Perch charge.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, meta, n_err = [], [], 0
    for i, fn in enumerate(files):
        if i % 200 == 0:
            print(f"  {i}/{len(files)}, {len(embeddings)} fenetres, {n_err} err, {time.time()-t0:.0f}s")
        if i > 0 and i % CHECKPOINT_EVERY == 0:
            save(embeddings, meta)
            print(f"  [checkpoint] {i} fichiers")
        try:
            audio = normalize_rms(librosa.load(str(sc_dir / fn), sr=SAMPLE_RATE, mono=True)[0])
            n_win = max(1, len(audio) // WINDOW_SAMPLES)
            for w in range(n_win):
                start = w * WINDOW_SAMPLES
                chunk = audio[start:start + WINDOW_SAMPLES]
                if len(chunk) < WINDOW_SAMPLES:
                    pad = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                    pad[:len(chunk)] = chunk
                    chunk = pad
                embeddings.append(extract_embedding(model, chunk))
                meta.append({"filename": fn, "start": w * 5, "end": w * 5 + 5})
        except Exception as e:
            n_err += 1
            if n_err <= 30:
                print(f"  [ERR] {fn}: {e}")

    print(f"Termine : {len(embeddings)} fenetres, {n_err} err, {(time.time()-t0)/60:.1f} min")
    save(embeddings, meta)
    print("Sauvegarde finale OK")


if __name__ == "__main__":
    main()
