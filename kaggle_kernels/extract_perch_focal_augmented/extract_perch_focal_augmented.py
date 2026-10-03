"""Extraction Perch v8 sur le focal AUGMENTE par bruit ambiant de soundscape.

Augmentation visant a reduire le domain gap : pendant l'extraction, chaque
fenetre focale est melangee a un segment ambiant tire d'un soundscape, pour
rapprocher le contenu d'entrainement des conditions de terrain.

Pour chaque fichier focal (tous, jusqu'a 3 fenetres/fichier, RMS<0.001 exclus) :
    fenetre_aug = normalize(focal) + ALPHA * normalize(bruit_soundscape)
puis renormalisation, puis embedding Perch 1280D.

Le bruit est tire d'un pool de NOISE_POOL segments 5 s pre-charges une fois
(rapide). Seed fixe pour reproductibilite.

Sortie : embeddings_perch_v8_focal_augmented.npy + _meta.csv (filename,
primary_label, window_idx). Comparable a embeddings_perch_v8_focal_full filtre
a window_idx<3 (memes fichiers, memes fenetres), pour mesurer l'effet de
l'augmentation a periemtre identique.
"""
import sys, time
from pathlib import Path
import librosa, numpy as np, pandas as pd
import tensorflow as tf, tensorflow_hub as hub

DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_PATH = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier/8"
OUTPUT_DIR = Path("/kaggle/working")
SR = 32_000
WS = SR * 5
MAX_WIN = 3
RMS_MIN = 0.001
ALPHA = 0.5          # niveau du bruit relatif au focal normalise
NOISE_POOL = 60      # nb de segments de bruit pre-charges
CHECKPOINT = 2000
rng = np.random.RandomState(42)


def rms(a): return float(np.sqrt(np.mean(a ** 2)))
def norm(a, t=0.1):
    r = rms(a); return a if r < 1e-6 else a * (t / r)


def build_noise_pool():
    sc_dir = DATA_ROOT / "train_soundscapes"
    files = sorted(p.name for p in sc_dir.glob("*.ogg"))
    chosen = rng.choice(len(files), size=min(NOISE_POOL, len(files)), replace=False)
    pool = []
    for i in chosen:
        try:
            a, _ = librosa.load(str(sc_dir / files[i]), sr=SR, mono=True)
            if len(a) >= WS:
                st = rng.randint(0, len(a) - WS + 1)
                pool.append(norm(a[st:st + WS].astype(np.float32)))
        except Exception:
            continue
    print(f"Pool de bruit : {len(pool)} segments")
    return pool


def augment(window, pool):
    noise = pool[rng.randint(len(pool))]
    return norm(norm(window) + ALPHA * noise)


def embed(model, w):
    out = model.infer_tf(tf.constant(w[None, :], dtype=tf.float32))
    if isinstance(out, dict):
        e = out.get("embedding")
        if e is None:
            e = next(v for k, v in out.items() if "embed" in k.lower())
    elif isinstance(out, (tuple, list)):
        e = out[1] if len(out) >= 2 else out[0]
    else:
        e = out
    return e.numpy()[0].astype(np.float32)


def save(emb, meta):
    np.save(OUTPUT_DIR / "embeddings_perch_v8_focal_augmented.npy", np.array(emb, dtype=np.float32))
    pd.DataFrame(meta).to_csv(OUTPUT_DIR / "embeddings_perch_v8_focal_augmented_meta.csv", index=False)


def main():
    t0 = time.time()
    for p in (DATA_ROOT, Path(MODEL_PATH)):
        if not p.exists():
            print("ERREUR path", p); sys.exit(1)
    df = pd.read_csv(DATA_ROOT / "train.csv")[["filename", "primary_label"]]
    model = hub.load(MODEL_PATH); print("Perch charge")
    pool = build_noise_pool()
    audio_root = DATA_ROOT / "train_audio"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb, meta, nerr, nsil = [], [], 0, 0
    for i, row in enumerate(df.itertuples()):
        if i % 500 == 0:
            print(f"  {i}/{len(df)}, {len(emb)} fenetres, {nsil} sil, {nerr} err, {time.time()-t0:.0f}s")
        if i > 0 and i % CHECKPOINT == 0:
            save(emb, meta); print(f"  [checkpoint] {i}")
        try:
            a, _ = librosa.load(str(audio_root / row.filename), sr=SR, mono=True)
            if rms(a) < RMS_MIN:
                nsil += 1; continue
            a = norm(a)
            for w in range(MAX_WIN):
                st = w * WS
                if st >= len(a): break
                chunk = a[st:st + WS]
                if len(chunk) < WS:
                    pad = np.zeros(WS, dtype=np.float32); pad[:len(chunk)] = chunk; chunk = pad
                emb.append(embed(model, augment(chunk.astype(np.float32), pool)))
                meta.append({"filename": row.filename, "primary_label": row.primary_label, "window_idx": w})
        except Exception as e:
            nerr += 1
            if nerr <= 30: print("  [ERR]", row.filename, e)
    print(f"Termine : {len(emb)} fenetres, {nsil} sil, {nerr} err, {(time.time()-t0)/60:.1f} min")
    save(emb, meta); print("Sauvegarde OK")


if __name__ == "__main__":
    main()
