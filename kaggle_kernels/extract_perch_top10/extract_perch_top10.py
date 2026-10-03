"""Extraction des embeddings Perch v8 sur le sous-ensemble BirdCLEF top10.

Reproduit exactement le perimetre du parquet local
data/processed/features_top10_n50_w3_all.parquet :
    - 10 especes top (les plus representees du primary_label)
    - 50 fichiers par espece, tirage seede (seed=42)
    - jusqu'a 3 fenetres de 5 s par fichier (les premieres)
    - audio 32 kHz mono, normalise RMS

Pour chaque fenetre, Perch v8 produit un embedding de dimension 1280. On
sauvegarde dans /kaggle/working/ :
    - embeddings_perch_v8_top10.npy : array (N_fenetres, 1280) float32
    - embeddings_perch_v8_top10_meta.csv : DataFrame avec colonnes
      filename, primary_label, window_idx, alignees a chaque ligne du npy.

Une fois le kernel termine, on telecharge ces deux fichiers en local et
on entraine LightGBM / LogReg dessus avec exactement le meme protocole
de CV 5 folds que les sessions 3.1 a 3.3.
"""
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


# --- Chemins Kaggle ---
# Quand le kernel est attache a une competition via competition_sources,
# le dataset est monte sous /kaggle/input/competitions/<slug>/.
DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
# Quand un modele est attache via model_sources, il est monte sous
# /kaggle/input/models/<owner>/<modelname>/<framework>/<variation>/<version>/.
# Path confirme par le log de la version 1 de ce kernel.
MODEL_PATH = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier/8"
OUTPUT_DIR = Path("/kaggle/working")

# --- Parametres audio (doivent correspondre a Perch v8) ---
SAMPLE_RATE = 32_000
WINDOW_DURATION = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 160 000

# --- Protocole sous-ensemble top10 ---
SEED = 42
N_TOP = 10
N_FILES = 50
N_WINDOWS = 3


def check_paths() -> None:
    """Verifie que les paths attendus existent. Sortie verbeuse en cas d'echec."""
    for path in (DATA_ROOT, Path(MODEL_PATH)):
        if not path.exists():
            print(f"ERREUR : chemin introuvable : {path}")
            print("Contenu /kaggle/input/ :")
            for p in sorted(Path("/kaggle/input").rglob("*"))[:200]:
                if p.is_dir() or p.suffix in {".pb", ".tflite", ".csv"}:
                    print(f"  {p}")
            sys.exit(1)


def normalize_rms(audio: np.ndarray, target: float = 0.1) -> np.ndarray:
    """Normalisation RMS, meme convention que src/audio.py local."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    return audio * (target / rms)


def window_audio(audio: np.ndarray, n_windows: int = N_WINDOWS) -> list[np.ndarray]:
    """Decoupe en n_windows fenetres 5 s (zero-pad la derniere si besoin)."""
    windows: list[np.ndarray] = []
    for i in range(n_windows):
        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        if start >= len(audio):
            break
        if end > len(audio):
            chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            available = audio[start:]
            chunk[: len(available)] = available
        else:
            chunk = audio[start:end].astype(np.float32)
        windows.append(chunk)
        if end >= len(audio):
            break
    return windows


def load_perch_model(model_path: str):
    """Charge Perch v8 depuis le SavedModel TensorFlow Hub."""
    print(f"Chargement Perch v8 depuis {model_path}")
    model = hub.load(model_path)
    print("Modele charge.")
    return model


def extract_embedding(model, window: np.ndarray) -> np.ndarray:
    """Forward pass d'une fenetre 5 s dans Perch. Retourne un vecteur 1280D.

    L'API Perch v8 attend un tensor (batch, samples) float32 et expose une
    methode infer_tf qui renvoie un dict {'embedding': ..., 'logits': ...}
    ou un tuple (logits, embedding) selon la version. On gere les deux cas.
    """
    win_tensor = tf.constant(window[None, :], dtype=tf.float32)
    output = model.infer_tf(win_tensor)
    if isinstance(output, dict):
        embedding = output.get("embedding")
        if embedding is None:
            # Fallback : chercher une cle qui ressemble.
            embedding = next(v for k, v in output.items() if "embed" in k.lower())
    elif isinstance(output, (tuple, list)):
        # Convention historique : (logits, embeddings)
        embedding = output[1] if len(output) >= 2 else output[0]
    else:
        embedding = output
    return embedding.numpy()[0].astype(np.float32)


def main() -> None:
    t_start = time.time()

    # 1. Verifier l'environnement et charger le modele
    check_paths()
    model = load_perch_model(MODEL_PATH)

    # 2. Charger les metadonnees et selectionner top10 x 50 fichiers
    print("Chargement train.csv")
    df = pd.read_csv(DATA_ROOT / "train.csv")[["primary_label", "filename"]]
    counts = df["primary_label"].value_counts()
    top_species = counts.head(N_TOP).index.tolist()
    print(f"Top {N_TOP} especes : {top_species}")

    rng = np.random.RandomState(SEED)
    parts = []
    for sp in top_species:
        subset = df[df["primary_label"] == sp]
        take = min(N_FILES, len(subset))
        chosen = rng.choice(subset.index.values, size=take, replace=False)
        parts.append(subset.loc[chosen])
    sampled = pd.concat(parts, ignore_index=True)
    print(f"Fichiers selectionnes : {len(sampled)}")

    # 3. Boucle d'extraction
    audio_root = DATA_ROOT / "train_audio"
    all_embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []
    n_errors = 0
    t_extract = time.time()

    for i, row in enumerate(sampled.itertuples()):
        if i % 25 == 0:
            elapsed = time.time() - t_extract
            print(f"  {i}/{len(sampled)} fichiers, "
                  f"{len(all_embeddings)} fenetres extraites, "
                  f"{n_errors} erreurs, {elapsed:.0f}s ecoules")

        audio_path = audio_root / row.filename
        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            audio = normalize_rms(audio)
            windows = window_audio(audio)
            for w_idx, win in enumerate(windows):
                emb = extract_embedding(model, win)
                all_embeddings.append(emb)
                meta_rows.append({
                    "filename": row.filename,
                    "primary_label": row.primary_label,
                    "window_idx": w_idx,
                })
        except Exception as err:
            n_errors += 1
            print(f"  [ERREUR] {row.filename}: {err}")

    extract_duration = time.time() - t_extract
    print(f"\nExtraction terminee : {len(all_embeddings)} embeddings, "
          f"{n_errors} erreurs, duree {extract_duration:.0f}s "
          f"({extract_duration / max(len(all_embeddings), 1) * 1000:.0f} ms/fenetre)")

    # 4. Sauvegarde
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    out_npy = OUTPUT_DIR / "embeddings_perch_v8_top10.npy"
    out_csv = OUTPUT_DIR / "embeddings_perch_v8_top10_meta.csv"
    np.save(out_npy, embeddings_arr)
    meta_df.to_csv(out_csv, index=False)

    print(f"\nSauvegarde :")
    print(f"  {out_npy} : shape {embeddings_arr.shape}, dtype {embeddings_arr.dtype}")
    print(f"  {out_csv} : shape {meta_df.shape}")

    print(f"\n=== Termine en {(time.time() - t_start):.0f}s ===")


if __name__ == "__main__":
    main()
