"""Shared constants and helpers used by analysis.py and make_figures.py.

Keeping the feature loading and the cross validation folds in one place
guarantees that every script sees the same clips, labels and folds.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SEED = 0
N_FOLDS = 5

SPECIES_GROUP = {
    "HumpbackWhale": "Mysticete",
    "BowheadWhale": "Mysticete",
    "NorthernRightWhale": "Mysticete",
    "KillerWhale": "Odontocete",
    "Narwhal": "Odontocete",
    "Beluga_WhiteWhale": "Odontocete",
    "BeardedSeal": "Pinniped",
    "RingedSeal": "Pinniped",
    "Walrus": "Pinniped",
}

# Column order written by extract_features.classical():
# MFCC means, MFCC SDs, delta MFCC means, delta MFCC SDs,
# then mean and SD of each spectral descriptor.
FEATURE_COLUMNS = (
    [f"mfccMean{i}" for i in range(20)]
    + [f"mfccStd{i}" for i in range(20)]
    + [f"deltaMfccMean{i}" for i in range(20)]
    + [f"deltaMfccStd{i}" for i in range(20)]
    + [f"{name}{stat}"
       for name in ["centroid", "bandwidth", "rolloff", "zcr", "rms"]
       for stat in ["Mean", "Std"]]
)
META_COLUMNS = ["species", "year", "filename", "durationSeconds", "originalSr"]


def load_features(feat_dir):
    """Load every feat/<species>.npz file written by extract_features.py.

    Returns
    -------
    features : DataFrame, one row of 90 acoustic features per clip
    mel : uint8 array (clips, 64, 128), log mel spectrograms
    meta : DataFrame with species, taxonomic group and recording session

    Clips with any non finite feature are dropped from all three outputs.
    """
    feature_blocks, mel_blocks, meta_blocks = [], [], []
    for path in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        feature_blocks.append(data["X"])
        mel_blocks.append(data["mel"])
        meta_blocks.append(pd.DataFrame(data["meta"].tolist(), columns=META_COLUMNS))

    features = pd.DataFrame(np.vstack(feature_blocks), columns=FEATURE_COLUMNS)
    mel = np.concatenate(mel_blocks)
    meta = pd.concat(meta_blocks, ignore_index=True)
    meta["group"] = meta["species"].map(SPECIES_GROUP)
    # Watkins file names start with a two digit year and a three digit tape
    # number; clips sharing those five characters come from the same tape.
    meta["session"] = meta["species"] + "_" + meta["filename"].str[:5]

    keep = np.isfinite(features.values).all(axis=1)
    features = features[keep].reset_index(drop=True)
    meta = meta[keep].reset_index(drop=True)
    return features, mel[keep], meta


def make_folds(features, labels, groups, grouped=True, n_splits=N_FOLDS):
    """Stratified 5 fold splits, grouped by recording session or random.

    With grouped=True no recording session contributes clips to both the
    training and the test side of a fold, which removes session leakage.
    """
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(splitter.split(features, labels, groups))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(features, labels))
