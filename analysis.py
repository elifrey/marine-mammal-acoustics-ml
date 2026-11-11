"""
Marine mammal call classification on 4,761 Watkins recordings (v1.1.0).

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

    EPOCHS=25 python analysis.py [feat_dir]

Writes figures/ and outputs/ (model_summary.csv holds every score).
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score, accuracy_score,
                             balanced_accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(0); torch.manual_seed(0); torch.set_num_threads(2)
FEAT_DIR = sys.argv[1] if len(sys.argv) > 1 else "feat"
FIG, OUT = "figures", "outputs"
os.makedirs(FIG, exist_ok=True); os.makedirs(OUT, exist_ok=True)

SPECIES_GROUP = {
    "HumpbackWhale": "Mysticete", "BowheadWhale": "Mysticete", "NorthernRightWhale": "Mysticete",
    "KillerWhale": "Odontocete", "Narwhal": "Odontocete", "Beluga_WhiteWhale": "Odontocete",
    "BeardedSeal": "Pinniped", "RingedSeal": "Pinniped", "Walrus": "Pinniped",
}
COLS = ([f"mfccMean{i}" for i in range(20)] + [f"mfccStd{i}" for i in range(20)] +
        [f"deltaMfccMean{i}" for i in range(20)] + [f"deltaMfccStd{i}" for i in range(20)] +
        [f"{n}{s}" for n in ["centroid", "bandwidth", "rolloff", "zcr", "rms"] for s in ["Mean", "Std"]])

# ---------------------------------------------------------------- load
Xs, Ms, metas = [], [], []
for f in sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz"))):
    d = np.load(f, allow_pickle=True)
    Xs.append(d["X"]); Ms.append(d["mel"]); metas.append(pd.DataFrame(d["meta"].tolist(),
        columns=["species", "year", "filename", "durationSeconds", "originalSr"]))
X = pd.DataFrame(np.vstack(Xs), columns=COLS)
mel = np.concatenate(Ms).astype(np.float32) / 255 * 8 - 4
meta = pd.concat(metas, ignore_index=True)
meta["group"] = meta["species"].map(SPECIES_GROUP)
# Watkins file names start with a two digit year and three digit tape number;
# clips sharing those five characters come from the same recording tape.
meta["session"] = meta["species"] + "_" + meta["filename"].str[:5]
keep = np.isfinite(X.values).all(axis=1)
X, mel, meta = X[keep].reset_index(drop=True), mel[keep], meta[keep].reset_index(drop=True)
print(f"{len(meta)} clips, {meta['species'].nunique()} species, {meta['session'].nunique()} recording sessions")
meta.to_csv(f"{OUT}/clip_metadata.csv", index=False)
counts = meta.groupby("species").agg(clips=("filename", "size"), sessions=("session", "nunique"))
counts.to_csv(f"{OUT}/clips_per_species.csv")
print(counts)

species_order = sorted(meta["species"].unique())
group_order = sorted(meta["group"].unique())
groups = meta["session"].values

def splitter(y, grouped=True, n=5):
    if grouped:
        return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0).split(X, y, groups))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=0).split(X, y))

def plot_cm(cm, labels, title, path, cmap="Blues"):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6)); plt.imshow(cmn, cmap=cmap, vmin=0, vmax=1); plt.colorbar(label="Row proportion")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                     color="white" if cmn[i, j] > 0.5 else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

results = []

# ---------------------------------------------------------------- unsupervised
Xsd = StandardScaler().fit_transform(X)
Xp = PCA(n_components=10, random_state=0).fit_transform(Xsd)
sil = {}
for k in range(2, 13):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xp)
    if np.bincount(lab).min() >= 10:
        sil[k] = silhouette_score(Xp, lab, sample_size=3000, random_state=0)
best_k = max(sil, key=sil.get)
clusters = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(Xp)
ari_sp = adjusted_rand_score(meta["species"], clusters); ari_gr = adjusted_rand_score(meta["group"], clusters)
print(f"KMeans k={best_k}: ARI species {ari_sp:.3f}, group {ari_gr:.3f}")
plt.figure(figsize=(6, 4)); plt.plot(list(sil), list(sil.values()), marker="o"); plt.axvline(best_k, ls="--", c="grey")
plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title("Silhouette by number of clusters"); plt.tight_layout()
plt.savefig(f"{FIG}/01_silhouette_by_k.png", dpi=150); plt.close()
plt.figure(figsize=(7, 5))
for g in group_order:
    i = meta["group"].values == g; plt.scatter(Xp[i, 0], Xp[i, 1], s=6, alpha=0.5, label=g)
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of acoustic features by taxonomic group")
plt.tight_layout(); plt.savefig(f"{FIG}/03_pca_by_group.png", dpi=150); plt.close()

# ---------------------------------------------------------------- random forest
def rf_cv(y, labels, name, prefix, grouped=True):
    y = np.asarray(y); folds = splitter(y, grouped)
    accs, baccs, f1s, cm = [], [], [], np.zeros((len(labels), len(labels)), int)
    imp = np.zeros(X.shape[1]); preds = np.empty(len(y), dtype=object)
    for tr, te in folds:
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=2, random_state=0)
        clf.fit(X.iloc[tr], y[tr]); p = clf.predict(X.iloc[te]); preds[te] = p
        accs.append(accuracy_score(y[te], p)); baccs.append(balanced_accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro")); cm += confusion_matrix(y[te], p, labels=labels)
        imp += clf.feature_importances_ / len(folds)
    tag = "grouped" if grouped else "random"
    if grouped:
        plot_cm(cm, labels, f"{name}: random forest, 5 fold CV grouped by recording session", f"{FIG}/{prefix}_rf_confusion.png")
        with open(f"{OUT}/{prefix}_rf_report.txt", "w") as fh:
            fh.write(classification_report(y, preds, labels=labels))
        fi = pd.Series(imp, index=X.columns).sort_values(ascending=False); fi.to_csv(f"{OUT}/{prefix}_rf_feature_importance.csv")
        plt.figure(figsize=(6, 6)); fi.head(15)[::-1].plot(kind="barh"); plt.title(f"{name}: top 15 features")
        plt.tight_layout(); plt.savefig(f"{FIG}/{prefix}_rf_feature_importance.png", dpi=150); plt.close()
    r = dict(task=name, model="random forest", cv=f"5 fold, {tag}", accuracy=np.mean(accs), accuracySd=np.std(accs),
             balancedAccuracy=np.mean(baccs), balancedAccuracySd=np.std(baccs), macroF1=np.mean(f1s))
    print(r); results.append(r); return r

t0 = time.time()
rf_cv(meta["species"], species_order, "Species (9 way)", "05_species", grouped=True)
rf_cv(meta["species"], species_order, "Species (9 way)", "05_species", grouped=False)
rf_cv(meta["group"], group_order, "Taxonomic group (3 way)", "06_group", grouped=True)
rf_cv(meta["group"], group_order, "Taxonomic group (3 way)", "06_group", grouped=False)
print(f"RF done in {time.time()-t0:.0f}s", flush=True)

for task, y in [("Species (9 way)", meta["species"]), ("Taxonomic group (3 way)", meta["group"])]:
    vc = y.value_counts(normalize=True)
    results.append(dict(task=task, model="majority class baseline", cv="", accuracy=vc.max(), accuracySd=0,
                        balancedAccuracy=1 / len(vc), balancedAccuracySd=0, macroF1=np.nan))

# ---------------------------------------------------------------- transformer
class SpectrogramTransformer(nn.Module):
    def __init__(self, n_mels=64, n_frames=128, pf=8, pt=8, d=64, heads=4, layers=2, n_classes=3, drop=0.3):
        super().__init__(); self.pf, self.pt = pf, pt; n_p = (n_mels // pf) * (n_frames // pt)
        self.embed = nn.Linear(pf * pt, d); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_p + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers); self.drop = nn.Dropout(drop); self.head = nn.Linear(d, n_classes)
    def forward(self, x):
        b = x.shape[0]
        x = x.unfold(1, self.pf, self.pf).unfold(2, self.pt, self.pt).contiguous().view(b, -1, self.pf * self.pt)
        t = torch.cat([self.cls.expand(b, -1, -1), self.embed(x)], 1) + self.pos
        return self.head(self.drop(self.enc(t)[:, 0]))

def augment(xb):
    # light SpecAugment: one frequency mask and one time mask per clip
    xb = xb.clone()
    for i in range(xb.shape[0]):
        f0 = np.random.randint(0, 64 - 8); xb[i, f0:f0 + np.random.randint(0, 8)] = 0
        t0_ = np.random.randint(0, 128 - 16); xb[i, :, t0_:t0_ + np.random.randint(0, 16)] = 0
    return xb

def transformer_cv(y, labels, name, prefix, epochs):
    y = np.asarray(y); lab2i = {c: i for i, c in enumerate(labels)}; yi = np.array([lab2i[v] for v in y])
    folds = splitter(y, grouped=True); accs, baccs, f1s = [], [], []
    cm = np.zeros((len(labels), len(labels)), int); curves = []
    for k, (tr, te) in enumerate(folds):
        torch.manual_seed(k)
        model = SpectrogramTransformer(n_classes=len(labels))
        w = torch.tensor(len(tr) / (len(labels) * np.bincount(yi[tr], minlength=len(labels)).clip(1)), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        dl = DataLoader(TensorDataset(torch.tensor(mel[tr]), torch.tensor(yi[tr])), batch_size=64, shuffle=True)
        Xte = torch.tensor(mel[te]); curve = []
        for ep in range(epochs):
            model.train()
            for xb, yb in dl:
                opt.zero_grad(); loss = crit(model(augment(xb)), yb); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                p = torch.cat([model(Xte[i:i + 256]).argmax(1) for i in range(0, len(Xte), 256)]).numpy()
            curve.append(balanced_accuracy_score(yi[te], p))
        curves.append(curve)
        accs.append(accuracy_score(yi[te], p)); baccs.append(balanced_accuracy_score(yi[te], p))
        f1s.append(f1_score(yi[te], p, average="macro")); cm += confusion_matrix(yi[te], p, labels=range(len(labels)))
        print(f"  {name} fold {k+1}: acc {accs[-1]:.3f}, balanced {baccs[-1]:.3f}", flush=True)
    plot_cm(cm, labels, f"{name}: transformer, 5 fold CV grouped by recording session", f"{FIG}/{prefix}_transformer_confusion.png", "Purples")
    plt.figure(figsize=(6, 4))
    for k, c in enumerate(curves): plt.plot(c, alpha=0.7, label=f"fold {k+1}")
    plt.xlabel("Epoch"); plt.ylabel("Held out balanced accuracy"); plt.title(f"{name}: transformer training"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/{prefix}_transformer_curves.png", dpi=150); plt.close()
    r = dict(task=name, model="transformer (from scratch, log mel)", cv="5 fold, grouped",
             accuracy=np.mean(accs), accuracySd=np.std(accs), balancedAccuracy=np.mean(baccs),
             balancedAccuracySd=np.std(baccs), macroF1=np.mean(f1s))
    print(r, flush=True); results.append(r)

EPOCHS = int(os.environ.get("EPOCHS", "25"))
t0 = time.time()
transformer_cv(meta["group"], group_order, "Taxonomic group (3 way)", "06_group", EPOCHS)
print(f"transformer group done in {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(results).to_csv(f"{OUT}/model_summary.csv", index=False)
if os.environ.get("SPECIES_TRANSFORMER", "1") == "1":
    transformer_cv(meta["species"], species_order, "Species (9 way)", "05_species", EPOCHS)
pd.DataFrame(results).to_csv(f"{OUT}/model_summary.csv", index=False)
json.dump(dict(clips=len(meta), sessions=int(meta["session"].nunique()), kmeansK=int(best_k),
               ariSpecies=ari_sp, ariGroup=ari_gr), open(f"{OUT}/run_info.json", "w"), indent=1)
print("Done.")
