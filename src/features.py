"""Extraction de features audio à partir du signal temporel."""
import librosa
import numpy as np

from src.config import SAMPLE_RATE


# --- Hyperparamètres globaux de l'extraction ---

# Coefficients MFCC : 13 pour la baseline (historique reconnaissance vocale),
# 40 pour la version riche du Jour 2 (couvre toute la gamme audio < 16 kHz, utile
# pour les chants aigus d'oiseaux et insectes).
N_MFCC_BASELINE = 13
N_MFCC_RICH = 40
N_MEL_BANDS = 128

# Statistiques agrégées appliquées à toute série temporelle (1 valeur par frame
# STFT). 5 stats : centralité, dispersion, extrêmes, médiane. On évite skewness
# et kurtosis qui sont bruitées sur peu de frames.
STAT_NAMES = ("mean", "std", "min", "max", "p50")


# --- Utilitaire de stats ---

def aggregate_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    """Agrège une série temporelle 1D en 5 statistiques scalaires.

    Toute famille de features audio produit naturellement une série temporelle
    (une valeur par frame STFT). Pour avoir un vecteur de taille fixe par
    fenêtre 5 s, on agrège chaque série avec ces 5 statistiques.

    Args:
        x: série 1D des valeurs par frame.
        prefix: préfixe à mettre devant chaque nom de stat (ex: "temp_rms").

    Returns:
        Dict {prefix_mean, prefix_std, prefix_min, prefix_max, prefix_p50}.
        Si x est vide ou contient des NaN, les NaN sont ignorés via np.nan*.
    """
    if x.size == 0:
        return {f"{prefix}_{s}": 0.0 for s in STAT_NAMES}

    return {
        f"{prefix}_mean": float(np.nanmean(x)),
        f"{prefix}_std": float(np.nanstd(x)),
        f"{prefix}_min": float(np.nanmin(x)),
        f"{prefix}_max": float(np.nanmax(x)),
        f"{prefix}_p50": float(np.nanmedian(x)),
    }


# --- Famille 1 : features temporelles ---

def temporal_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Features extraites directement du signal temporel (pas de transformation fréquentielle).

    Ces features sont peu coûteuses à calculer et capturent la dynamique
    temporelle générale du signal : niveau sonore, variations rapides
    (passages par zéro), caractère impulsif vs continu.

    Familles produites :
    - RMS (root mean square) : niveau sonore moyen par frame. Distingue
      les passages forts (cri) des passages faibles (silence ambiant).
    - ZCR (zero crossing rate) : nombre de changements de signe par frame.
      Proxy grossier de la fréquence dominante : un sifflement aigu a un
      ZCR élevé, un grondement basse fréquence un ZCR faible.
    - Pic d'amplitude global : max |y|. Détecte les bursts.
    - Crest factor : pic / RMS global. Caractérise les sons impulsifs
      (cri bref = crest élevé) vs continus (chant prolongé = crest bas).
    - Active ratio : fraction de frames RMS au-dessus de la médiane.
      Approxime la densité d'activité dans la fenêtre.

    Args:
        y: signal audio 1D.
        sr: sample rate (utilisé pour ZCR cohérent).

    Returns:
        Dict d'environ 13 features préfixées 'temp_'.
    """
    features: dict[str, float] = {}

    # RMS par frame, puis agrégation. librosa découpe en frames de 2048 ech.
    # par défaut avec hop 512.
    rms = librosa.feature.rms(y=y).flatten()
    features.update(aggregate_stats(rms, "temp_rms"))

    # ZCR par frame.
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    features.update(aggregate_stats(zcr, "temp_zcr"))

    # Scalaires globaux : peak, crest factor, active ratio.
    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    rms_global = float(np.sqrt(np.mean(y ** 2)))
    features["temp_peak"] = peak
    features["temp_crest_factor"] = peak / rms_global if rms_global > 0 else 0.0

    median_rms = float(np.median(rms))
    features["temp_active_ratio"] = float(np.mean(rms > median_rms)) if rms.size else 0.0

    return features


# --- Famille 2 : features spectrales ---

def spectral_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Features extraites du spectre de magnitude (STFT).

    Le spectre est calculé une fois et passé à toutes les sous-fonctions
    librosa qui le demandent, pour éviter de refaire la STFT 6 fois.

    Familles produites :
    - Centroïde spectral : "centre de gravité" du spectre. Indique où se
      concentre l'énergie. Sons aigus = centroïde haut.
    - Bandwidth spectrale : écart-type pondéré autour du centroïde. Mesure
      l'étalement du spectre.
    - Rolloff (0.85 et 0.50) : fréquence en dessous de laquelle se trouve
      X% de l'énergie. Deux niveaux pour capturer la forme du spectre.
    - Flatness : ratio moyenne géométrique / arithmétique du spectre.
      0 = signal très tonal (sinusoïde pure), 1 = bruit blanc.
    - Contraste spectral : différence entre pics et vallées dans 7 sous-bandes.
      Très utilisé en bioacoustique pour distinguer chants harmoniques vs
      bruits diffus.
    - Flux spectral : variation du spectre d'une frame à l'autre. Élevé
      pour des sons modulés (chant rapide), faible pour des sons stables.

    Args:
        y: signal audio 1D.
        sr: sample rate.

    Returns:
        Dict d'environ 65 features préfixées 'spec_'.
    """
    features: dict[str, float] = {}

    # STFT magnitude unique, partagée par toutes les features qui en ont besoin.
    S = np.abs(librosa.stft(y))

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).flatten()
    features.update(aggregate_stats(centroid, "spec_centroid"))

    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).flatten()
    features.update(aggregate_stats(bandwidth, "spec_bandwidth"))

    rolloff_85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).flatten()
    features.update(aggregate_stats(rolloff_85, "spec_rolloff85"))

    rolloff_50 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.50).flatten()
    features.update(aggregate_stats(rolloff_50, "spec_rolloff50"))

    flatness = librosa.feature.spectral_flatness(S=S).flatten()
    features.update(aggregate_stats(flatness, "spec_flatness"))

    # Contraste spectral : 7 bandes par défaut (n_bands=6, +1 sous-bande basse).
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for band_idx in range(contrast.shape[0]):
        features.update(aggregate_stats(contrast[band_idx], f"spec_contrast_{band_idx:02d}"))

    # Flux spectral : différence L2 normalisée entre frames consécutives.
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    features.update(aggregate_stats(flux, "spec_flux"))

    return features


# --- Famille 3 : MFCC complets (40 MFCC + delta + delta2) ---

def mfcc_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """MFCC riches avec dérivées première et seconde.

    Voir docstring de extract_mfcc_baseline pour le pipeline interne. Ici on
    monte à 40 coefficients (au lieu de 13) et on ajoute les dérivées
    temporelles delta et delta2, qui capturent respectivement la vitesse et
    l'accélération de variation du timbre frame à frame.

    Pour chaque série (40 MFCC + 40 delta + 40 delta2 = 120 séries), on
    agrège avec les 5 stats standard. Total : 600 features.

    Args:
        y: signal audio 1D.
        sr: sample rate.

    Returns:
        Dict de 600 features préfixées 'mfcc_NN_', 'mfcc_delta_NN_',
        'mfcc_delta2_NN_'.
    """
    features: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_RICH)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(N_MFCC_RICH):
        features.update(aggregate_stats(mfcc[i], f"mfcc_{i:02d}"))
        features.update(aggregate_stats(delta[i], f"mfcc_delta_{i:02d}"))
        features.update(aggregate_stats(delta2[i], f"mfcc_delta2_{i:02d}"))

    return features


# --- Famille 4 : statistiques du mel-spectrogramme ---

def mel_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Statistiques bande par bande du mel-spectrogramme.

    Le mel-spectrogramme est l'étape juste avant la DCT dans la chaîne MFCC.
    Garder ses 128 bandes "brutes" donne des features complémentaires aux
    MFCC : les MFCC capturent l'ENVELOPPE globale, les bandes mel capturent
    le DÉTAIL fréquentiel par sous-bande.

    Pour chaque bande on prend juste mean + std (2 stats au lieu de 5) pour
    éviter d'exploser le nombre total : 128 * 5 = 640 ferait doublon avec
    les MFCC. 128 * 2 = 256 reste raisonnable.

    Args:
        y: signal audio 1D.
        sr: sample rate.

    Returns:
        Dict de 256 features préfixées 'mel_NNN_mean' et 'mel_NNN_std'.
    """
    features: dict[str, float] = {}

    # Conversion en dB (échelle log perceptuelle) : valeurs en dB plus stables
    # numériquement que les énergies brutes, et plus discriminantes.
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL_BANDS)
    mel_db = librosa.power_to_db(mel_spec)

    for band in range(N_MEL_BANDS):
        features[f"mel_{band:03d}_mean"] = float(np.mean(mel_db[band]))
        features[f"mel_{band:03d}_std"] = float(np.std(mel_db[band]))

    return features


# --- Famille 5 : indices bioacoustiques ---

# Bandes de fréquences (Hz) pour les indices écologiques. Conventions issues de
# la littérature soundscape ecology (Pijanowski et al. 2011, Sueur et al. 2008).
ANTHRO_BAND_HZ = (1000, 2000)   # bruits humains : moteurs, vent, infrasons
BIO_BAND_HZ = (2000, 8000)      # gamme vocale typique des oiseaux


def _band_indices(freqs: np.ndarray, low: float, high: float) -> tuple[int, int]:
    """Retourne les indices de freqs (issus de librosa.fft_frequencies) qui
    correspondent à la bande [low, high] Hz."""
    return int(np.searchsorted(freqs, low)), int(np.searchsorted(freqs, high))


def bioacoustic_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """5 indices bioacoustiques calculés à partir du spectrogramme.

    Implémentation manuelle (sans scikit-maad) pour pouvoir défendre chaque
    formule. Les indices sont scalaires : 1 valeur par fenêtre.

    Args:
        y: signal audio 1D.
        sr: sample rate.

    Returns:
        Dict de 5 features : bio_aci, bio_adi, bio_aei, bio_bi, bio_ndsi.
    """
    S = np.abs(librosa.stft(y))  # magnitude spectrogram (n_freq, n_frames)
    freqs = librosa.fft_frequencies(sr=sr)
    n_freq, n_frames = S.shape

    # ACI : somme sur les bandes des variations frame-à-frame, normalisée par
    # la somme totale de la bande. Mesure la "modulation" du signal.
    if n_frames > 1:
        deltas = np.abs(np.diff(S, axis=1))
        sums = np.sum(S[:, :-1], axis=1)
        aci_per_band = np.where(sums > 0, np.sum(deltas, axis=1) / sums, 0.0)
        aci = float(np.sum(aci_per_band))
    else:
        aci = 0.0

    # Pour ADI/AEI : on divise le spectre en 10 sous-bandes égales et on calcule
    # la proportion d'énergie au-dessus d'un seuil (-50 dB) dans chacune.
    # Convention soundscape ecology.
    energy_db = librosa.amplitude_to_db(S)
    threshold_db = -50.0
    n_subbands = 10
    band_edges = np.linspace(0, n_freq, n_subbands + 1, dtype=int)
    active_fractions = []
    for k in range(n_subbands):
        sub = energy_db[band_edges[k]:band_edges[k + 1]]
        active_fractions.append(float(np.mean(sub > threshold_db)) if sub.size else 0.0)
    p = np.array(active_fractions)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / n_subbands

    # ADI = entropie de Shannon de la distribution p.
    adi = float(-np.sum(p * np.log(p + 1e-12)))

    # AEI = indice de Gini = somme des |diff| de p triée, normalisée.
    p_sorted = np.sort(p)
    cum = np.cumsum(p_sorted)
    aei = float(1.0 - 2.0 * np.sum(cum) / n_subbands + 1.0 / n_subbands)

    # BI : énergie moyenne dans la bande biologique 2-8 kHz, normalisée.
    bio_lo, bio_hi = _band_indices(freqs, *BIO_BAND_HZ)
    bio_energy = np.mean(S[bio_lo:bio_hi]) if bio_hi > bio_lo else 0.0
    bi = float(bio_energy)

    # NDSI : (bio - anthro) / (bio + anthro), où on prend la moyenne d'énergie
    # par bande. Borne entre -1 (tout anthropique) et +1 (tout biologique).
    anthro_lo, anthro_hi = _band_indices(freqs, *ANTHRO_BAND_HZ)
    anthro_energy = float(np.mean(S[anthro_lo:anthro_hi])) if anthro_hi > anthro_lo else 0.0
    total = bi + anthro_energy
    ndsi = float((bi - anthro_energy) / total) if total > 0 else 0.0

    return {
        "bio_aci": aci,
        "bio_adi": adi,
        "bio_aei": aei,
        "bio_bi": bi,
        "bio_ndsi": ndsi,
    }


# --- Famille 6 : pitch et chroma ---

def pitch_chroma_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Fréquence fondamentale (pitch) et distribution chromatique.

    Pitch (via librosa.pyin) : détecte la fréquence fondamentale frame par
    frame. Pour les sons tonals (chants harmoniques), elle suit la mélodie.
    Pour les sons non tonals (bruit, percussions), elle retourne NaN.

    Chroma : projette le spectre sur 12 classes de hauteur tonale
    (C, C#, ..., B) sans considérer l'octave. Utile pour caractériser
    la "couleur tonale" d'un chant.

    Args:
        y: signal audio 1D.
        sr: sample rate.

    Returns:
        Dict d'environ 30 features préfixées 'pitch_' et 'chroma_'.
    """
    features: dict[str, float] = {}

    # Pitch via pyin. Bornes 50-2000 Hz : couvre la gamme vocale des oiseaux
    # avec marge. Plus haut serait possible mais pyin devient instable.
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr, fmin=50, fmax=2000, frame_length=2048
        )
        f0_clean = f0[~np.isnan(f0)]
        if f0_clean.size > 0:
            features.update(aggregate_stats(f0_clean, "pitch_f0"))
        else:
            features.update({f"pitch_f0_{s}": 0.0 for s in STAT_NAMES})
        features["pitch_voiced_ratio"] = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0
    except Exception:
        # pyin peut échouer sur certains signaux dégénérés.
        features.update({f"pitch_f0_{s}": 0.0 for s in STAT_NAMES})
        features["pitch_voiced_ratio"] = 0.0

    # Chroma : 12 classes de hauteur, agrégées en mean + std (2 stats x 12 = 24).
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    note_names = ["C", "Csh", "D", "Dsh", "E", "F", "Fsh", "G", "Gsh", "A", "Ash", "B"]
    for i, note in enumerate(note_names):
        features[f"chroma_{note}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{note}_std"] = float(np.std(chroma[i]))

    return features


# --- Fonction principale d'extraction (assemblage des 6 familles) ---

# Mapping famille -> fonction. Permet de sélectionner des sous-ensembles via
# l'argument families de extract_features (mode "fast" ou "ablation").
FEATURE_FAMILIES = {
    "temporal": temporal_features,
    "spectral": spectral_features,
    "mfcc": mfcc_features,
    "mel": mel_features,
    "bioacoustic": bioacoustic_features,
    "pitch_chroma": pitch_chroma_features,
}


def extract_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    families: list[str] | None = None,
) -> dict[str, float]:
    """Extrait toutes les familles de features sur un signal (fenêtre 5 s).

    Fonction principale du module : appelle chaque famille demandée et
    concatène les dictionnaires. C'est elle qu'on appellera depuis le script
    d'extraction massive (Session 2.5).

    Args:
        y: signal audio 1D, longueur quelconque mais typiquement WINDOW_SAMPLES.
        sr: sample rate.
        families: liste de noms de familles à calculer (clés de FEATURE_FAMILIES).
            Si None, toutes les familles sont calculées. Ordre suit FEATURE_FAMILIES.

    Returns:
        Dict plat de ~970 features si toutes les familles.
    """
    if families is None:
        families = list(FEATURE_FAMILIES.keys())

    all_features: dict[str, float] = {}
    for name in families:
        if name not in FEATURE_FAMILIES:
            raise ValueError(f"Famille inconnue: {name}. Disponibles: {list(FEATURE_FAMILIES)}")
        all_features.update(FEATURE_FAMILIES[name](y, sr))

    return all_features


# --- Baseline du Jour 1 (conservée pour reproductibilité) ---

def extract_mfcc_baseline(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Calcule moyenne et écart-type des 13 premiers MFCC sur une fenêtre.

    Les MFCC (Mel-Frequency Cepstral Coefficients) sont la représentation
    historique en bioacoustique. La chaîne de calcul effectuée en interne
    par librosa.feature.mfcc est : signal -> STFT -> spectre de puissance
    -> filtre mel (échelle perceptuelle) -> log -> DCT. Le résultat est un
    vecteur (n_mfcc, n_frames) où chaque colonne décrit l'enveloppe
    spectrale d'une frame de quelques millisecondes.

    Pour avoir un vecteur de taille fixe par fenêtre 5 s, on agrège sur le
    temps avec la moyenne (timbre dominant) et l'écart-type (variabilité
    temporelle). Ce sont les agrégations les plus simples et les plus
    informatives pour démarrer ; on diversifiera au Jour 2.

    Args:
        y: signal audio 1D float32, longueur quelconque (typiquement
            WINDOW_SAMPLES = 160 000).
        sr: sample rate du signal.

    Returns:
        Dictionnaire plat de 26 features. Noms : mfcc_00_mean, mfcc_00_std,
        mfcc_01_mean, ..., mfcc_12_std.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC_BASELINE)

    features: dict[str, float] = {}
    for i in range(N_MFCC_BASELINE):
        features[f"mfcc_{i:02d}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i:02d}_std"] = float(np.std(mfcc[i]))

    return features
