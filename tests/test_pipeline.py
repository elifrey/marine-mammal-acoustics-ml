"""Fast checks of the parts the study's conclusions depend on.

Run from the repository root with:  pytest
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import SpectrogramTransformer, mel_to_float, spec_augment  # noqa: E402
from common import FEATURE_COLUMNS, SPECIES_GROUP, load_features, make_folds  # noqa: E402
from extract_features import N_FRAMES, N_MELS, TARGET_SR, classical, mel  # noqa: E402


@pytest.fixture
def toy_labels():
    """60 clips of 3 classes recorded on 12 tapes (5 clips per tape)."""
    rng = np.random.default_rng(0)
    sessions = np.repeat([f"tape{i}" for i in range(12)], 5)
    labels = np.repeat(["a", "b", "c"], 20)
    return rng.normal(size=(60, 4)), labels, sessions


def test_grouped_folds_never_share_a_session(toy_labels):
    features, labels, sessions = toy_labels
    for train, test in make_folds(features, labels, sessions, grouped=True):
        assert not set(sessions[train]) & set(sessions[test])


def test_random_folds_do_share_sessions(toy_labels):
    """The leakage the paper measures: random splits put clips of one tape on both sides."""
    features, labels, sessions = toy_labels
    shared = [set(sessions[train]) & set(sessions[test])
              for train, test in make_folds(features, labels, sessions, grouped=False)]
    assert any(shared)


def test_folds_cover_every_clip_once(toy_labels):
    features, labels, sessions = toy_labels
    for grouped in (True, False):
        test_rows = np.concatenate([test for _, test in make_folds(features, labels, sessions, grouped)])
        assert sorted(test_rows) == list(range(len(labels)))


def test_feature_extraction_shapes():
    y = np.random.default_rng(1).normal(size=TARGET_SR).astype(np.float32) * 0.1
    assert classical(y).shape == (len(FEATURE_COLUMNS),) == (90,)
    spectrogram = mel(y)
    assert spectrogram.shape == (N_MELS, N_FRAMES)
    assert spectrogram.dtype == np.uint8


def test_mel_round_trip_range():
    restored = mel_to_float(np.array([0, 255], dtype=np.uint8))
    assert restored.min() == pytest.approx(-4) and restored.max() == pytest.approx(4)


def test_load_features_drops_non_finite_rows_and_sets_sessions(tmp_path):
    features = np.ones((3, 90), np.float32)
    features[1, 0] = np.nan
    meta = [["Walrus", "1970", f"70{i:03d}01.wav", 1.0, 48000] for i in range(3)]
    np.savez_compressed(tmp_path / "Walrus.npz", X=features, mel=np.zeros((3, 64, 128), np.uint8),
                        meta=np.array(meta, dtype=object))
    loaded, spectrograms, info = load_features(str(tmp_path))
    assert len(loaded) == len(spectrograms) == len(info) == 2
    assert list(info["session"]) == ["Walrus_70000", "Walrus_70002"]
    assert set(info["group"]) == {SPECIES_GROUP["Walrus"]}
    assert isinstance(loaded, pd.DataFrame) and list(loaded.columns) == FEATURE_COLUMNS


def test_transformer_output_shape_and_augment_keeps_input():
    batch = torch.randn(4, 64, 128)
    original = batch.clone()
    augmented = spec_augment(batch)
    assert torch.equal(batch, original)  # augmentation works on a copy
    assert augmented.shape == batch.shape
    logits = SpectrogramTransformer(n_classes=9)(augmented)
    assert logits.shape == (4, 9)
