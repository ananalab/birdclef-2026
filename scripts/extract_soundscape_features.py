import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.audio import load_audio, normalize_rms
from src.config import DATA_PROCESSED, DATASET_DIR, SAMPLE_RATE, WINDOW_SAMPLES
from src.features import extract_features

LABELS_CSV = DATASET_DIR / "train_soundscapes_labels.csv"
SC_DIR = DATASET_DIR / "train_soundscapes"
OUTPUT = DATA_PROCESSED / "features_soundscapes_heldout.parquet"


def to_seconds(hms: str) -> int:
    h, m, s = (int(x) for x in str(hms).split(":"))
    return h * 3600 + m * 60 + s


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


def process_file(filename: str, rows: list[dict]) -> list[dict]:
    """Charge un soundscape une fois, extrait les features de ses fenetres annotees."""
    try:
        signal = normalize_rms(load_audio(SC_DIR / filename))
    except Exception:
        return []
    out = []
    for r in rows:
        try:
            feats = extract_features(get_window_at(signal, to_seconds(r["start"])))
        except Exception:
            continue
        feats["filename"] = filename
        feats["start"] = r["start"]
        feats["end"] = r["end"]
        feats["primary_label"] = r["primary_label"]
        out.append(feats)
    return out


def main() -> None:
    t0 = time.time()
    labels = pd.read_csv(LABELS_CSV)
    print(f"{len(labels)} fenetres annotees, {labels['filename'].nunique()} soundscapes")

    by_file = {}
    for r in labels.to_dict(orient="records"):
        by_file.setdefault(r["filename"], []).append(r)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_file)(fn, rows) for fn, rows in by_file.items()
    )
    all_rows = [r for sub in results for r in sub]
    df = pd.DataFrame(all_rows)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="snappy")
    n_feat = df.shape[1] - 4
    print(f"Ecrit {OUTPUT} : {df.shape}, {n_feat} features, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
