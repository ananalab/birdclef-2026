"""Extraction Perch v8 sur les soundscapes annotes (jeu held-out).

Lit train_soundscapes_labels.csv (1478 fenetres 5 s annotees sur 66
soundscapes de terrain), extrait pour chaque fenetre l'embedding Perch
1280D. Sert a l'evaluation honnete des deux pistes en conditions reelles
(in situ Pantanal, multi-label, jamais vu en entrainement).

La colonne primary_label du CSV est multi-label (especes separees par ';').
On la conserve telle quelle dans la meta de sortie ; l'evaluation multi-label
se fera en local.

Sortie /kaggle/working/ :
    - embeddings_perch_v8_soundscapes.npy : (N, 1280) float32
    - embeddings_perch_v8_soundscapes_meta.csv : filename, start, end,
      primary_label (aligne ligne a ligne avec le npy)
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


def to_seconds(hms: str) -> int:
    """'00:01:05' -> 65 (secondes)."""
    h, m, s = (int(x) for x in str(hms).split(":"))
    return h * 3600 + m * 60 + s


def normalize_rms(audio: np.ndarray, target: float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    return audio * (target / rms)


def get_window_at(audio: np.ndarray, start_sec: int) -> np.ndarray:
    start = start_sec * SAMPLE_RATE
    end = start + WINDOW_SAMPLES
    if start >= len(audio):
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if end > len(audio):
        chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        chunk[: len(audio) - start] = audio[start:]
        return chunk
    return audio[start:end].astype(np.float32)


def extract_embedding(model, window: np.ndarray) -> np.ndarray:
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


def main() -> None:
    t_start = time.time()
    for path in (DATA_ROOT, Path(MODEL_PATH)):
        if not path.exists():
            print(f"ERREUR : path introuvable {path}")
            sys.exit(1)

    labels = pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")
    print(f"{len(labels)} fenetres annotees, {labels['filename'].nunique()} soundscapes")

    model = hub.load(MODEL_PATH)
    print("Perch charge.")

    sc_root = DATA_ROOT / "train_soundscapes"
    embeddings = []
    meta_rows = []
    n_errors = 0
    # On charge chaque soundscape une seule fois (plusieurs fenetres par fichier).
    cache_audio = {}
    t0 = time.time()
    for i, row in enumerate(labels.itertuples()):
        if i % 200 == 0:
            print(f"  {i}/{len(labels)}, {len(embeddings)} fenetres, {time.time()-t0:.0f}s")
        fn = row.filename
        try:
            if fn not in cache_audio:
                cache_audio.clear()  # 1 seul fichier en cache (RAM)
                audio, _ = librosa.load(str(sc_root / fn), sr=SAMPLE_RATE, mono=True)
                cache_audio[fn] = normalize_rms(audio)
            window = get_window_at(cache_audio[fn], to_seconds(row.start))
            emb = extract_embedding(model, window)
            embeddings.append(emb)
            meta_rows.append({
                "filename": fn,
                "start": row.start,
                "end": row.end,
                "primary_label": row.primary_label,
            })
        except Exception as err:
            n_errors += 1
            print(f"  [ERREUR] {fn} @ {row.start}: {err}")

    print(f"Termine : {len(embeddings)} fenetres, {n_errors} erreurs, {time.time()-t0:.0f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "embeddings_perch_v8_soundscapes.npy",
            np.array(embeddings, dtype=np.float32))
    pd.DataFrame(meta_rows).to_csv(
        OUTPUT_DIR / "embeddings_perch_v8_soundscapes_meta.csv", index=False)
    print(f"Sauvegarde OK. Total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
