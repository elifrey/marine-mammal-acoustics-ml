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
import glob
import os

import librosa
import numpy as np
import soundfile as sf

TARGET_SR, N_MFCC, N_MELS, N_FRAMES = 22050, 20, 64, 128


def _pool(feature):
    """Mean and SD over time for every row of a (n_features, n_frames) array."""
    return np.concatenate([feature.mean(axis=1), feature.std(axis=1)])


def classical(y):
    """90 classical acoustic features for one clip at TARGET_SR."""
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    # delta needs an odd window no longer than the clip
    width = min(9, mfcc.shape[1])
    width -= (width % 2 == 0)
    delta = np.zeros_like(mfcc) if width < 3 else librosa.feature.delta(mfcc, width=width)
    parts = [mfcc, delta, librosa.feature.spectral_centroid(y=y, sr=TARGET_SR),
             librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR),
             librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR),
             librosa.feature.zero_crossing_rate(y=y), librosa.feature.rms(y=y)]
    # order matches analysis.py columns: mfcc means, mfcc SDs, delta means, delta SDs,
    # then mean and SD of each spectral descriptor
    return np.concatenate([_pool(p) for p in parts]).astype(np.float32)


def mel(y):
    """64 x 128 log mel spectrogram, z scored within the clip and stored as uint8."""
    power = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(power, ref=np.max)
    pad = max(0, N_FRAMES - log_mel.shape[1])
    log_mel = np.pad(log_mel, ((0, 0), (0, pad)), constant_values=log_mel.min())[:, :N_FRAMES]
    z = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return np.clip(np.round((np.clip(z, -4, 4) + 4) / 8 * 255), 0, 255).astype(np.uint8)


def load_clip(path):
    """Read a wav file, mix to mono, resample to TARGET_SR and pad very short clips."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    y = data.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_mq")
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    return y, sr


def extract_species(species_dir, out_path):
    """Features, spectrograms and metadata for every clip of one species."""
    species = os.path.basename(os.path.dirname(species_dir))
    features, mels, meta, unreadable = [], [], [], []
    for path in sorted(glob.glob(os.path.join(species_dir, "*", "*.wav"))):
        try:
            y, sr = load_clip(path)
            features.append(classical(y))
            mels.append(mel(y))
            meta.append([species, os.path.basename(os.path.dirname(path)), os.path.basename(path),
                         len(y) / TARGET_SR, sr])
        except Exception as e:  # keep going; unreadable files are logged in the output
            unreadable.append([path, str(e)[:80]])
    np.savez_compressed(out_path, X=np.stack(features), mel=np.stack(mels),
                        meta=np.array(meta, dtype=object), bad=np.array(unreadable, dtype=object))
    print(out_path, len(meta), "clips,", len(unreadable), "unreadable", flush=True)


def main():
    os.makedirs("feat", exist_ok=True)
    for species_dir in sorted(glob.glob("data_raw/*/")):
        species = os.path.basename(os.path.dirname(species_dir))
        out_path = f"feat/{species}.npz"
        if not os.path.exists(out_path):  # rerunning skips species already done
            extract_species(species_dir, out_path)

if __name__ == "__main__":
    main()
