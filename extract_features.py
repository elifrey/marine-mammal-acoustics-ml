"""
Extract the two inputs used by analysis.py from every clip in data_raw/:

1. 90 classical acoustic features per clip (20 MFCCs and their deltas, mean
   and standard deviation over time, plus spectral centroid, bandwidth,
   rolloff, zero crossing rate and RMS loudness, mean and SD).
2. A 64 x 128 log mel spectrogram per clip (first 128 frames, about 1.5 s at
   22,050 Hz with a 256 sample hop), z scored within the clip, clipped to
   +/-4 SD and stored as uint8 to keep the feature files small.

Every recording is mixed to mono and resampled to 22,050 Hz first, because
the Watkins clips were digitised at many different sample rates.

    python extract_features.py          # writes feat/<species>.npz
"""
import glob, os
import numpy as np, soundfile as sf, librosa

TARGET_SR, N_MFCC, N_MELS, N_FRAMES = 22050, 20, 64, 128

def _pool(f):
    return np.concatenate([f.mean(axis=1), f.std(axis=1)])

def classical(y):
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    w = min(9, mfcc.shape[1]); w -= (w % 2 == 0)
    d = np.zeros_like(mfcc) if w < 3 else librosa.feature.delta(mfcc, width=w)
    parts = [mfcc, d, librosa.feature.spectral_centroid(y=y, sr=TARGET_SR),
             librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR),
             librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR),
             librosa.feature.zero_crossing_rate(y=y), librosa.feature.rms(y=y)]
    # order matches analysis.py columns: mfcc means, mfcc SDs, delta means, delta SDs,
    # then mean and SD of each spectral descriptor
    return np.concatenate([_pool(p) for p in parts]).astype(np.float32)

def mel(y):
    m = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=1024, hop_length=256)
    l = librosa.power_to_db(m, ref=np.max)
    l = np.pad(l, ((0, 0), (0, max(0, N_FRAMES - l.shape[1]))), constant_values=l.min())[:, :N_FRAMES]
    l = (l - l.mean()) / (l.std() + 1e-6)
    return np.clip(np.round((np.clip(l, -4, 4) + 4) / 8 * 255), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    os.makedirs("feat", exist_ok=True)
    for sp_dir in sorted(glob.glob("data_raw/*/")):
        sp = os.path.basename(os.path.dirname(sp_dir)); out = f"feat/{sp}.npz"
        if os.path.exists(out): continue
        F, M, meta, bad = [], [], [], []
        for path in sorted(glob.glob(os.path.join(sp_dir, "*", "*.wav"))):
            try:
                data, sr = sf.read(path, always_2d=True, dtype="float32"); y = data.mean(axis=1)
                if sr != TARGET_SR: y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_mq")
                if len(y) < 2048: y = np.pad(y, (0, 2048 - len(y)))
                F.append(classical(y)); M.append(mel(y))
                meta.append([sp, os.path.basename(os.path.dirname(path)), os.path.basename(path), len(y) / TARGET_SR, sr])
            except Exception as e:
                bad.append([path, str(e)[:80]])
        np.savez_compressed(out, X=np.stack(F), mel=np.stack(M), meta=np.array(meta, dtype=object),
                            bad=np.array(bad, dtype=object))
        print(out, len(meta), "clips,", len(bad), "unreadable", flush=True)
