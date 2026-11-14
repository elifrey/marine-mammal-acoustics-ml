"""
Marine mammal call classification on 4,761 Watkins recordings.

Inputs: feat/*.npz written by extract_features.py (90 classical acoustic
features plus a 64 x 128 log mel spectrogram per clip).

Runs, in order:
  1. Unsupervised check: PCA + KMeans, adjusted Rand index against labels.
  2. Random forest (400 trees, class weighted) for species (9 way) and
     taxonomic group (3 way), each evaluated twice with 5 fold CV:
     grouped by recording session (no tape on both sides) and random.
  3. Majority class baselines.
  4. A small vision style transformer trained from scratch on the log mel
     spectrograms, same grouped folds, class weighted loss, SpecAugment.

    python analysis.py [feat_dir] [--epochs 25] [--skip-species-transformer]

Writes figures/ and outputs/ (model_summary.csv holds every score).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import FIG_DIR, OUT_DIR, RF_TREES, SEED, load_features, make_folds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

# transformer training settings
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-2
FREQ_MASK, TIME_MASK = 8, 16  # maximum SpecAugment mask widths (mel bands, frames)


# ---------------------------------------------------------------- helpers

def mel_to_float(mel_uint8: np.ndarray) -> np.ndarray:
    """Undo the uint8 storage of extract_features.mel(): back to z scores in [-4, 4]."""
    return mel_uint8.astype(np.float32) / 255 * 8 - 4


def summarise_scores(task: str, model: str, cv: str, accs: list[float], baccs: list[float],
                     f1s: list[float]) -> dict:
    """One row of outputs/model_summary.csv."""
    return dict(task=task, model=model, cv=cv,
                accuracy=np.mean(accs), accuracySd=np.std(accs),
                balancedAccuracy=np.mean(baccs), balancedAccuracySd=np.std(baccs),
                macroF1=np.mean(f1s))


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: str, cmap: str = "Blues") -> None:
    """Confusion matrix coloured by row proportion, annotated with counts."""
    row_share = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label="Row proportion")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if row_share[i, j] > 0.5 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- 1. unsupervised check

def unsupervised_check(features: pd.DataFrame, meta: pd.DataFrame, group_order: list[str]) -> dict:
    """PCA to 10 components, KMeans with k chosen by silhouette, ARI against labels."""
    projected = PCA(n_components=10, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(features))

    silhouette = {}
    for k in range(2, 13):
        clusters = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(projected)
        if np.bincount(clusters).min() >= 10:  # skip solutions with tiny clusters
            silhouette[k] = silhouette_score(projected, clusters, sample_size=3000, random_state=SEED)
    best_k = max(silhouette, key=silhouette.get)
    clusters = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(projected)
    ari_species = adjusted_rand_score(meta["species"], clusters)
    ari_group = adjusted_rand_score(meta["group"], clusters)
    print(f"KMeans k={best_k}: ARI species {ari_species:.3f}, group {ari_group:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(list(silhouette), list(silhouette.values()), marker="o")
    plt.axvline(best_k, ls="--", c="grey")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette by number of clusters")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_silhouette_by_k.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for group in group_order:
        rows = meta["group"].values == group
        plt.scatter(projected[rows, 0], projected[rows, 1], s=6, alpha=0.5, label=group)
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of acoustic features by taxonomic group")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_pca_by_group.png", dpi=150)
    plt.close()

    return dict(kmeansK=int(best_k), ariSpecies=ari_species, ariGroup=ari_group)


# ---------------------------------------------------------------- 2. random forest

def random_forest_cv(features: pd.DataFrame, y, groups: np.ndarray, labels: list[str], task: str, prefix: str,
                     grouped: bool = True) -> dict:
    """5 fold random forest; the grouped run also writes reports and figures."""
    y = np.asarray(y)
    folds = make_folds(features, y, groups, grouped)
    accs, baccs, f1s = [], [], []
    cm = np.zeros((len(labels), len(labels)), int)
    importance = np.zeros(features.shape[1])
    predictions = np.empty(len(y), dtype=object)

    for train, test in folds:
        clf = RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced",
                                     n_jobs=2, random_state=SEED)
        clf.fit(features.iloc[train], y[train])
        pred = clf.predict(features.iloc[test])
        predictions[test] = pred
        accs.append(accuracy_score(y[test], pred))
        baccs.append(balanced_accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average="macro"))
        cm += confusion_matrix(y[test], pred, labels=labels)
        importance += clf.feature_importances_ / len(folds)

    if grouped:
        plot_confusion(cm, labels, f"{task}: random forest, 5 fold CV grouped by recording session",
                       f"{FIG_DIR}/{prefix}_rf_confusion.png")
        with open(f"{OUT_DIR}/{prefix}_rf_report.txt", "w") as fh:
            fh.write(classification_report(y, predictions, labels=labels))
        ranked = pd.Series(importance, index=features.columns).sort_values(ascending=False)
        ranked.to_csv(f"{OUT_DIR}/{prefix}_rf_feature_importance.csv")
        plt.figure(figsize=(6, 6))
        ranked.head(15)[::-1].plot(kind="barh")
        plt.title(f"{task}: top 15 features")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{prefix}_rf_feature_importance.png", dpi=150)
        plt.close()

    split_name = "grouped" if grouped else "random"
    row = summarise_scores(task, "random forest", f"5 fold, {split_name}", accs, baccs, f1s)
    print(row)
    return row


# ---------------------------------------------------------------- 3. baselines

def majority_baseline(task: str, y: pd.Series) -> dict:
    """Accuracy of always predicting the most common class."""
    shares = y.value_counts(normalize=True)
    return dict(task=task, model="majority class baseline", cv="",
                accuracy=shares.max(), accuracySd=0,
                balancedAccuracy=1 / len(shares), balancedAccuracySd=0, macroF1=np.nan)


# ---------------------------------------------------------------- 4. transformer

class SpectrogramTransformer(nn.Module):
    """Small vision style transformer on 8 x 8 patches of a 64 x 128 log mel spectrogram."""

    def __init__(self, n_mels: int = 64, n_frames: int = 128, patch_f: int = 8, patch_t: int = 8,
                 dim: int = 64, heads: int = 4, layers: int = 2, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.patch_f, self.patch_t = patch_f, patch_t
        n_patches = (n_mels // patch_f) * (n_frames // patch_t)
        self.embed = nn.Linear(patch_f * patch_t, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        patches = (x.unfold(1, self.patch_f, self.patch_f)
                    .unfold(2, self.patch_t, self.patch_t)
                    .contiguous()
                    .view(batch, -1, self.patch_f * self.patch_t))
        tokens = torch.cat([self.cls.expand(batch, -1, -1), self.embed(patches)], 1) + self.pos
        return self.head(self.dropout(self.encoder(tokens)[:, 0]))


def spec_augment(batch: torch.Tensor) -> torch.Tensor:
    """Light SpecAugment: one frequency mask and one time mask per clip."""
    batch = batch.clone()
    for i in range(batch.shape[0]):
        n_mels, n_frames = batch.shape[1], batch.shape[2]
        f0 = np.random.randint(0, n_mels - FREQ_MASK)
        batch[i, f0:f0 + np.random.randint(0, FREQ_MASK)] = 0
        t0 = np.random.randint(0, n_frames - TIME_MASK)
        batch[i, :, t0:t0 + np.random.randint(0, TIME_MASK)] = 0
    return batch


def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Class index predictions in batches, without gradients."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch_size]).argmax(1)
                          for i in range(0, len(x), batch_size)]).numpy()


def train_one_fold(mel: np.ndarray, y_index: np.ndarray, train: np.ndarray, test: np.ndarray,
                   n_classes: int, epochs: int, fold: int) -> tuple[np.ndarray, list[float]]:
    """Train on one fold; return test predictions and held out balanced accuracy per epoch."""
    torch.manual_seed(fold)
    model = SpectrogramTransformer(n_classes=n_classes)
    class_counts = np.bincount(y_index[train], minlength=n_classes).clip(1)
    weights = torch.tensor(len(train) / (n_classes * class_counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader = DataLoader(TensorDataset(torch.tensor(mel[train]), torch.tensor(y_index[train])),
                        batch_size=BATCH_SIZE, shuffle=True)
    x_test = torch.tensor(mel[test])

    curve = []
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(spec_augment(xb)), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        pred = predict(model, x_test)
        curve.append(balanced_accuracy_score(y_index[test], pred))
    return pred, curve


def transformer_cv(mel: np.ndarray, y, groups: np.ndarray, labels: list[str], task: str, prefix: str,
                   epochs: int) -> dict:
    """Grouped 5 fold CV of the transformer, with confusion matrix and learning curves."""
    y = np.asarray(y)
    label_index = {label: i for i, label in enumerate(labels)}
    y_index = np.array([label_index[v] for v in y])
    folds = make_folds(mel, y, groups, grouped=True)
    accs, baccs, f1s, curves = [], [], [], []
    cm = np.zeros((len(labels), len(labels)), int)

    for fold, (train, test) in enumerate(folds):
        pred, curve = train_one_fold(mel, y_index, train, test, len(labels), epochs, fold)
        curves.append(curve)
        accs.append(accuracy_score(y_index[test], pred))
        baccs.append(balanced_accuracy_score(y_index[test], pred))
        f1s.append(f1_score(y_index[test], pred, average="macro"))
        cm += confusion_matrix(y_index[test], pred, labels=range(len(labels)))
        print(f"  {task} fold {fold + 1}: acc {accs[-1]:.3f}, balanced {baccs[-1]:.3f}", flush=True)

    plot_confusion(cm, labels, f"{task}: transformer, 5 fold CV grouped by recording session",
                   f"{FIG_DIR}/{prefix}_transformer_confusion.png", "Purples")
    plt.figure(figsize=(6, 4))
    for fold, curve in enumerate(curves):
        plt.plot(curve, alpha=0.7, label=f"fold {fold + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Held out balanced accuracy")
    plt.title(f"{task}: transformer training")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{prefix}_transformer_curves.png", dpi=150)
    plt.close()

    row = summarise_scores(task, "transformer (from scratch, log mel)", "5 fold, grouped", accs, baccs, f1s)
    print(row, flush=True)
    return row


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("feat_dir", nargs="?", default="feat", help="folder with feat/<species>.npz")
    parser.add_argument("--epochs", type=int, default=25, help="transformer training epochs per fold")
    parser.add_argument("--skip-species-transformer", action="store_true",
                        help="train the transformer for taxonomic groups only (much faster)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch")
    return parser.parse_args()


def main(feat_dir: str = "feat", epochs: int = 25, species_transformer: bool = True, threads: int = 2) -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    features, mel_uint8, meta = load_features(feat_dir)
    mel = mel_to_float(mel_uint8)
    groups = meta["session"].values
    print(f"{len(meta)} clips, {meta['species'].nunique()} species, "
          f"{meta['session'].nunique()} recording sessions")
    meta.to_csv(f"{OUT_DIR}/clip_metadata.csv", index=False)
    counts = meta.groupby("species").agg(clips=("filename", "size"), sessions=("session", "nunique"))
    counts.to_csv(f"{OUT_DIR}/clips_per_species.csv")
    print(counts)

    species_order = sorted(meta["species"].unique())
    group_order = sorted(meta["group"].unique())
    tasks = [("Species (9 way)", "species", species_order, "05_species"),
             ("Taxonomic group (3 way)", "group", group_order, "06_group")]

    run_info = unsupervised_check(features, meta, group_order)

    results = []
    start = time.time()
    for task, column, labels, prefix in tasks:
        for grouped in (True, False):
            results.append(random_forest_cv(features, meta[column], groups, labels, task, prefix, grouped))
    print(f"RF done in {time.time() - start:.0f}s", flush=True)

    for task, column, _, _ in tasks:
        results.append(majority_baseline(task, meta[column]))

    def save_summary() -> None:
        pd.DataFrame(results).to_csv(f"{OUT_DIR}/model_summary.csv", index=False)

    start = time.time()
    results.append(transformer_cv(mel, meta["group"], groups, group_order,
                                  "Taxonomic group (3 way)", "06_group", epochs))
    print(f"transformer group done in {time.time() - start:.0f}s", flush=True)
    save_summary()
    if species_transformer:
        results.append(transformer_cv(mel, meta["species"], groups, species_order,
                                      "Species (9 way)", "05_species", epochs))
    save_summary()

    with open(f"{OUT_DIR}/run_info.json", "w") as fh:
        json.dump(dict(clips=len(meta), sessions=int(meta["session"].nunique()), **run_info), fh, indent=1)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args.feat_dir, args.epochs, not args.skip_species_transformer, args.threads)
