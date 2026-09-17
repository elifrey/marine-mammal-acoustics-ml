"""
Arctic and subarctic marine mammal call classification from bioacoustic recordings.

Pipeline:
  1. Load real WAV clips (Watkins Marine Mammal Sound Database subset).
  2. Extract classical acoustic features (MFCCs, spectral descriptors) per clip.
  3. Unsupervised: PCA plus KMeans clustering on classical features.
  4. Supervised: random forest classifiers (species, and broad taxonomic group).
  5. A small transformer (trained from scratch) on log mel spectrograms for the
     broad taxonomic group task, reported honestly against the random forest
     baseline on the same held out split.

All functions are imported by build_notebook.py so the notebook and this
script share one source of truth.
"""

import os
import glob
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, accuracy_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(0)

TARGET_SR = 22050
N_MFCC = 20
N_MELS = 64
N_FRAMES = 128  # fixed time axis length for spectrogram inputs

DATA_DIR = "data_raw/clips"
FIG_DIR = "figures"
OUT_DIR = "outputs"

# Broad taxonomic / functional group for each species, the biological
# analogue of the "era" grouping used in the seabird diet project: a coarse
# label the model is never told directly, only asked to recover from the
# acoustic signal.
SPECIES_GROUP = {
    "HumpbackWhale": "Mysticete (baleen whale)",
    "BowheadWhale": "Mysticete (baleen whale)",
    "NorthernRightWhale": "Mysticete (baleen whale)",
    "KillerWhale": "Odontocete (toothed whale)",
    "Narwhal": "Odontocete (toothed whale)",
    "Beluga_WhiteWhale": "Odontocete (toothed whale)",
    "BeardedSeal": "Pinniped (seal / walrus)",
    "RingedSeal": "Pinniped (seal / walrus)",
    "Walrus": "Pinniped (seal / walrus)",
}

# Genuinely Arctic or Subarctic species among the nine, for context in the
# writeup (not used as a model input).
ARCTIC_SPECIES = {
    "Narwhal", "Beluga_WhiteWhale", "BowheadWhale",
    "RingedSeal", "BeardedSeal", "Walrus",
}


# The individual clip extraction endpoint on the Internet Archive mirror of
# the Watkins Marine Mammal Sound Database: it streams one member out of the
# 6.2 gigabyte zip without downloading the whole archive.
ARCHIVE_ZIP_URL = (
    "https://ia600600.us.archive.org/view_archive.php"
    "?archive=/9/items/watkins-marine-mammal-sound-database-full-cuts/"
    "Watkins_Marine_Mammal_Sound_Database_full_cuts.zip"
)
ARCHIVE_INNER_PREFIX = "Watkins_Marine_Mammal_Sound_Database"


def ensure_clips_downloaded():
    """Fetch the 225 clips listed in data_raw/manifest.csv, one file at a time.

    The audio itself is not committed to this repository (see README and
    LICENSE_DATA.md): only the small manifest is, and this function
    re-derives the exact same subset directly from the public Internet
    Archive mirror the first time analysis.py is run. Already downloaded
    clips are left in place, so repeated runs only fetch what is missing.
    """
    manifest = pd.read_csv("data_raw/manifest.csv")
    missing = False
    for _, row in manifest.iterrows():
        out_path = os.path.join(DATA_DIR, row["species"], row["filename"])
        if not os.path.exists(out_path):
            missing = True
            break
    if not missing:
        return

    print("Downloading clips from the Internet Archive mirror (first run only)...")
    for _, row in manifest.iterrows():
        species_dir = os.path.join(DATA_DIR, row["species"])
        os.makedirs(species_dir, exist_ok=True)
        out_path = os.path.join(species_dir, row["filename"])
        if os.path.exists(out_path):
            continue
        remote_path = f"{ARCHIVE_INNER_PREFIX}/{row['species']}/{row['year']}/{row['filename']}"
        url = ARCHIVE_ZIP_URL + "&file=" + urllib.parse.quote(remote_path)
        urllib.request.urlretrieve(url, out_path)


def load_clips():
    """Load every WAV clip, resampled to a common sample rate, with metadata."""
    ensure_clips_downloaded()
    rows = []
    for species_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if not os.path.isdir(species_dir):
            continue
        species = os.path.basename(species_dir)
        for wav_path in sorted(glob.glob(os.path.join(species_dir, "*.wav"))):
            y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            rows.append({
                "species": species,
                "group": SPECIES_GROUP[species],
                "isArctic": species in ARCTIC_SPECIES,
                "filename": os.path.basename(wav_path),
                "path": wav_path,
                "durationSeconds": len(y) / TARGET_SR,
                "waveform": y,
            })
    meta = pd.DataFrame(rows)
    return meta


def _pool(feat):
    """Mean and standard deviation across time for a (features, frames) array."""
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])


def extract_classical_features(y):
    """One fixed length feature vector per clip: MFCCs, deltas, spectral shape."""
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    # very short clips yield few frames; delta's default width (9) needs an
    # odd width no larger than the number of frames available
    delta_width = min(9, mfcc.shape[1])
    if delta_width % 2 == 0:
        delta_width -= 1
    if delta_width < 3:
        dmfcc = np.zeros_like(mfcc)
    else:
        dmfcc = librosa.feature.delta(mfcc, width=delta_width)
    centroid = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    return np.concatenate([
        _pool(mfcc), _pool(dmfcc), _pool(centroid),
        _pool(bandwidth), _pool(rolloff), _pool(zcr), _pool(rms),
    ])


def build_feature_matrix(meta):
    feats = np.vstack([extract_classical_features(y) for y in meta["waveform"]])
    n_mfcc_names = [f"mfccMean{i}" for i in range(N_MFCC)] + [f"mfccStd{i}" for i in range(N_MFCC)]
    n_dmfcc_names = [f"deltaMfccMean{i}" for i in range(N_MFCC)] + [f"deltaMfccStd{i}" for i in range(N_MFCC)]
    other_names = []
    for name in ["centroid", "bandwidth", "rolloff", "zcr", "rms"]:
        other_names += [f"{name}Mean", f"{name}Std"]
    columns = n_mfcc_names + n_dmfcc_names + other_names
    return pd.DataFrame(feats, columns=columns)


def build_melspec_tensor(meta):
    """Fixed size log mel spectrogram per clip, stacked into one array."""
    specs = []
    for y in meta["waveform"]:
        mel = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=1024, hop_length=256,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        if logmel.shape[1] < N_FRAMES:
            pad = N_FRAMES - logmel.shape[1]
            logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant", constant_values=logmel.min())
        else:
            logmel = logmel[:, :N_FRAMES]
        # normalise each clip to zero mean unit variance for stable training
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        specs.append(logmel.astype(np.float32))
    return np.stack(specs)


# ---------------------------------------------------------------------------
# Unsupervised
# ---------------------------------------------------------------------------

def run_unsupervised(X, meta):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=10, random_state=0)
    Xp = pca.fit_transform(Xs)

    # Restrict the silhouette based search to solutions where every cluster
    # has a reasonable number of members, otherwise the score is maximised by
    # peeling off small outlier clusters rather than finding real structure.
    MIN_CLUSTER_SIZE = 10
    sil_scores = {}
    valid_ks = []
    for k in range(2, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xp)
        sizes = np.bincount(km.labels_)
        sil_scores[k] = silhouette_score(Xp, km.labels_)
        if sizes.min() >= MIN_CLUSTER_SIZE:
            valid_ks.append(k)
    candidates = valid_ks if valid_ks else list(sil_scores.keys())
    best_k = max(candidates, key=lambda k: sil_scores[k])

    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(Xp)
    meta = meta.copy()
    meta["cluster"] = km.labels_

    ari_species = adjusted_rand_score(meta["species"], meta["cluster"])
    ari_group = adjusted_rand_score(meta["group"], meta["cluster"])

    cluster_vs_group = pd.crosstab(meta["cluster"], meta["group"])
    cluster_vs_species = pd.crosstab(meta["cluster"], meta["species"])

    plt.figure(figsize=(6, 4))
    plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o")
    plt.axvline(best_k, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score by number of clusters")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_silhouette_by_k.png", dpi=150)
    plt.close()

    for colour_by, fname, title in [
        ("cluster", "02_pca_by_cluster.png", "PCA projection coloured by cluster"),
        ("group", "03_pca_by_group.png", "PCA projection coloured by taxonomic group"),
        ("species", "04_pca_by_species.png", "PCA projection coloured by species"),
    ]:
        plt.figure(figsize=(7, 5))
        categories = meta[colour_by].astype("category")
        for cat in categories.cat.categories:
            idx = categories == cat
            plt.scatter(Xp[idx.values, 0], Xp[idx.values, 1], label=str(cat), s=28, alpha=0.8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(title)
        plt.legend(fontsize=7, loc="best", ncol=1)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{fname}", dpi=150)
        plt.close()

    return meta, best_k, sil_scores, ari_species, ari_group, cluster_vs_species, cluster_vs_group


# ---------------------------------------------------------------------------
# Classical supervised (random forest)
# ---------------------------------------------------------------------------

def run_rf_classifier(X, y, label_name, fig_prefix, class_order=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0,
    )
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(clf, X, y, cv=cv)

    labels = class_order if class_order is not None else sorted(pd.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{label_name} confusion matrix, held out test set")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fig_prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(6, 6))
    fi.head(15)[::-1].plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title(f"{label_name}, top 15 features")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fig_prefix}_feature_importance.png", dpi=150)
    plt.close()

    return acc, cv_scores, report, fi, (X_train.index, X_test.index)


# ---------------------------------------------------------------------------
# Small transformer on mel spectrograms
# ---------------------------------------------------------------------------

class SpectrogramTransformer(nn.Module):
    def __init__(self, n_mels, n_frames, patch_f=8, patch_t=8, d_model=64,
                 n_heads=4, n_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        assert n_mels % patch_f == 0 and n_frames % patch_t == 0
        self.patch_f, self.patch_t = patch_f, patch_t
        n_patches = (n_mels // patch_f) * (n_frames // patch_t)
        patch_dim = patch_f * patch_t

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def patchify(self, x):
        # x: (batch, n_mels, n_frames)
        b, f, t = x.shape
        x = x.unfold(1, self.patch_f, self.patch_f).unfold(2, self.patch_t, self.patch_t)
        # -> (batch, f/patch_f, t/patch_t, patch_f, patch_t)
        x = x.contiguous().view(b, -1, self.patch_f * self.patch_t)
        return x

    def forward(self, x):
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        cls_out = self.dropout(encoded[:, 0])
        return self.head(cls_out)


def run_transformer_classifier(specs, labels, class_names, train_idx, test_idx, epochs=60):
    """Train a small transformer from scratch on log mel spectrograms.

    train_idx / test_idx are positional indices into specs/labels, matched to
    the same split used for the random forest group classifier so the two
    approaches are compared on identical held out clips.
    """
    label_to_int = {c: i for i, c in enumerate(class_names)}
    y_int = np.array([label_to_int[l] for l in labels])

    X_train = torch.tensor(specs[train_idx])
    y_train = torch.tensor(y_int[train_idx], dtype=torch.long)
    X_test = torch.tensor(specs[test_idx])
    y_test = torch.tensor(y_int[test_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = SpectrogramTransformer(
        n_mels=specs.shape[1], n_frames=specs.shape[2], n_classes=len(class_names),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    history = {"trainLoss": [], "testAcc": []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()
        history["trainLoss"].append(epoch_loss)
        history["testAcc"].append(test_acc)

    model.eval()
    with torch.no_grad():
        final_pred = model(X_test).argmax(dim=1).numpy()
    final_acc = accuracy_score(y_test.numpy(), final_pred)
    cm = confusion_matrix(y_test.numpy(), final_pred, labels=list(range(len(class_names))))

    plt.figure(figsize=(6, 4))
    plt.plot(history["trainLoss"], label="Training loss")
    plt.plot(history["testAcc"], label="Held out accuracy")
    plt.xlabel("Epoch")
    plt.title("Transformer training on log mel spectrograms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_transformer_training_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Purples")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=20, ha="right", fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Transformer group confusion matrix, held out test set")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_transformer_confusion_matrix.png", dpi=150)
    plt.close()

    return final_acc, history, cm


def example_spectrogram_figure(meta):
    """One example log mel spectrogram per taxonomic group, for the notebook."""
    hop_length = 256
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, group in zip(axes, sorted(meta["group"].unique())):
        row = meta[meta["group"] == group].iloc[0]
        mel = librosa.feature.melspectrogram(y=row["waveform"], sr=TARGET_SR, n_mels=N_MELS, n_fft=1024, hop_length=hop_length)
        logmel = librosa.power_to_db(mel, ref=np.max)
        n_frames = logmel.shape[1]
        duration = n_frames * hop_length / TARGET_SR
        img = ax.imshow(logmel, origin="lower", aspect="auto", cmap="magma",
                         extent=[0, duration, 0, N_MELS])
        ax.set_title(f"{group}\n({row['species']})", fontsize=9)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Mel bin")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/00_example_spectrograms.png", dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = load_clips()
    print(f"Loaded {len(meta)} clips across {meta['species'].nunique()} species "
          f"and {meta['group'].nunique()} taxonomic groups")

    example_spectrogram_figure(meta)

    X = build_feature_matrix(meta)
    X.to_csv(f"{OUT_DIR}/classical_features.csv", index=False)
    meta.drop(columns=["waveform"]).to_csv(f"{OUT_DIR}/clip_metadata.csv", index=False)

    (clustered_meta, best_k, sil_scores, ari_species, ari_group,
     cluster_vs_species, cluster_vs_group) = run_unsupervised(X, meta)
    print(f"Unsupervised: chosen k = {best_k}, "
          f"ARI vs species = {ari_species:.3f}, ARI vs group = {ari_group:.3f}")
    clustered_meta.drop(columns=["waveform"]).to_csv(f"{OUT_DIR}/cluster_assignments.csv", index=False)
    cluster_vs_species.to_csv(f"{OUT_DIR}/cluster_vs_species.csv")
    cluster_vs_group.to_csv(f"{OUT_DIR}/cluster_vs_group.csv")

    species_order = sorted(meta["species"].unique())
    acc_sp, cv_sp, report_sp, fi_sp, _ = run_rf_classifier(
        X, meta["species"], "Species (9 way)", "05_species", class_order=species_order,
    )
    print(f"Species classifier: held out accuracy {acc_sp:.3f}, "
          f"5 fold CV {cv_sp.mean():.3f} +/- {cv_sp.std():.3f}")
    with open(f"{OUT_DIR}/species_classifier_report.txt", "w") as f:
        f.write(report_sp)
    fi_sp.to_csv(f"{OUT_DIR}/species_feature_importance.csv")

    group_order = sorted(meta["group"].unique())
    acc_gr, cv_gr, report_gr, fi_gr, (train_idx, test_idx) = run_rf_classifier(
        X, meta["group"], "Taxonomic group (3 way)", "06_group", class_order=group_order,
    )
    print(f"Group classifier: held out accuracy {acc_gr:.3f}, "
          f"5 fold CV {cv_gr.mean():.3f} +/- {cv_gr.std():.3f}")
    with open(f"{OUT_DIR}/group_classifier_report.txt", "w") as f:
        f.write(report_gr)
    fi_gr.to_csv(f"{OUT_DIR}/group_feature_importance.csv")

    majority_baseline = meta["group"].value_counts(normalize=True).max()

    print("Building log mel spectrograms for the transformer...")
    specs = build_melspec_tensor(meta)
    # positional indices matching the pandas index used by the RF split
    pos_train = [meta.index.get_loc(i) for i in train_idx]
    pos_test = [meta.index.get_loc(i) for i in test_idx]
    acc_transformer, history, cm_transformer = run_transformer_classifier(
        specs, meta["group"].values, group_order, pos_train, pos_test, epochs=60,
    )
    print(f"Transformer group classifier: held out accuracy {acc_transformer:.3f} "
          f"(random forest on the same split: {acc_gr:.3f}, majority baseline: {majority_baseline:.3f})")

    summary = pd.DataFrame([{
        "task": "species (9 way, random forest)", "heldOutAccuracy": acc_sp,
        "cvMeanAccuracy": cv_sp.mean(), "cvStdAccuracy": cv_sp.std(),
    }, {
        "task": "taxonomic group (3 way, random forest)", "heldOutAccuracy": acc_gr,
        "cvMeanAccuracy": cv_gr.mean(), "cvStdAccuracy": cv_gr.std(),
    }, {
        "task": "taxonomic group (3 way, transformer on spectrograms)",
        "heldOutAccuracy": acc_transformer, "cvMeanAccuracy": np.nan, "cvStdAccuracy": np.nan,
    }])
    summary.to_csv(f"{OUT_DIR}/model_summary.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
