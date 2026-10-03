import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src.audio import load_audio, normalize_rms
from src.config import DATA_PROCESSED, DATASET_DIR, SAMPLE_RATE, TRAIN_AUDIO_DIR, WINDOW_SAMPLES
from src.features import extract_features

REF_PARQUET = DATA_PROCESSED / "features_focal_full_w3_all.parquet"
SC_DIR = DATASET_DIR / "train_soundscapes"
SHARDS = DATA_PROCESSED / "_shards_features_aug"
OUT = DATA_PROCESSED / "features_focal_augmented.parquet"
META_COLS = ("filename", "primary_label", "window_idx", "secondary_labels")
ALPHA = 0.5
NOISE_POOL = 60
rng = np.random.RandomState(42)


def rms(a): return float(np.sqrt(np.mean(a ** 2)))
def norm(a, t=0.1):
    r = rms(a); return a if r < 1e-6 else a * (t / r)


def get_window(audio, w):
    s = w * WINDOW_SAMPLES; e = s + WINDOW_SAMPLES
    if s >= len(audio): return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if e > len(audio):
        c = np.zeros(WINDOW_SAMPLES, dtype=np.float32); c[:len(audio)-s] = audio[s:]; return c
    return audio[s:e].astype(np.float32)


def build_noise_pool():
    files = sorted(p.name for p in SC_DIR.glob("*.ogg"))
    idx = rng.choice(len(files), size=min(NOISE_POOL, len(files)), replace=False)
    pool = []
    for i in idx:
        try:
            a = load_audio(SC_DIR / files[i])
            if len(a) >= WINDOW_SAMPLES:
                st = rng.randint(0, len(a) - WINDOW_SAMPLES + 1)
                pool.append(norm(a[st:st+WINDOW_SAMPLES].astype(np.float32)))
        except Exception:
            continue
    return pool


def proc_file(fn, label, wins, pool, seed):
    r = np.random.RandomState(seed)
    try:
        sig = norm(load_audio(TRAIN_AUDIO_DIR / fn))
    except Exception:
        return []
    out = []
    for w in wins:
        win = get_window(sig, w)
        aug = norm(norm(win) + ALPHA * pool[r.randint(len(pool))])
        try:
            f = extract_features(aug)
        except Exception:
            continue
        f["filename"] = fn; f["primary_label"] = label; f["window_idx"] = w
        out.append(f)
    return out


def main():
    t0 = time.time()
    ref = pd.read_parquet(REF_PARQUET, columns=["filename", "primary_label", "window_idx"])
    grouped = ref.groupby(["primary_label", "filename"])["window_idx"].apply(list)
    species = sorted(ref["primary_label"].unique())
    print(f"{len(species)} especes, {ref['filename'].nunique()} fichiers, {len(ref)} fenetres")
    pool = build_noise_pool()
    print(f"Pool de bruit : {len(pool)} segments")
    SHARDS.mkdir(parents=True, exist_ok=True)

    for si, sp in enumerate(species, 1):
        shard = SHARDS / f"{sp}.parquet"
        if shard.exists():
            # Verifier que le shard est lisible (un fichier ecrit a moitie au
            # moment d'une mise en veille serait corrompu : on le refait).
            try:
                pd.read_parquet(shard, columns=["window_idx"])
                print(f"  ({si}/{len(species)}) {sp} deja fait"); continue
            except Exception:
                print(f"  ({si}/{len(species)}) {sp} shard corrompu, re-extraction")
                shard.unlink(missing_ok=True)
        files = grouped.loc[sp]
        res = Parallel(n_jobs=-1)(
            delayed(proc_file)(fn, sp, wins, pool, hash(fn) % (2**31))
            for fn, wins in files.items()
        )
        rows = [x for sub in res for x in sub]
        if rows:
            pd.DataFrame(rows).to_parquet(shard, index=False)
        print(f"  ({si}/{len(species)}) {sp}: {len(rows)} fenetres | total {(time.time()-t0)/60:.0f} min")

    parts = [pd.read_parquet(s) for s in sorted(SHARDS.glob("*.parquet"))]
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT, index=False)
    print(f"Ecrit {OUT}: {full.shape} en {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
