"""
Publication figures for v1.1.0. Run after analysis.py:
    python make_figures.py [feat_dir]
Reads outputs/model_summary.csv and outputs/clip_metadata.csv, recomputes the
random forest out of fold predictions (same folds and seeds as analysis.py)
for per species scores, and writes figures/paper_*.png (300 dpi) plus
outputs/rf_per_species_f1.csv.
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

FEAT_DIR = sys.argv[1] if len(sys.argv) > 1 else "feat"
FIG, OUT = "figures", "outputs"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
GROUP_C = {"Mysticete": C1, "Odontocete": C2, "Pinniped": C3}
GROUP_NAME = {"Mysticete": "Baleen whales", "Odontocete": "Toothed whales", "Pinniped": "Seals and walrus"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
NICE = {"HumpbackWhale": "Humpback whale", "BowheadWhale": "Bowhead whale", "NorthernRightWhale": "N. right whale",
        "KillerWhale": "Killer whale", "Narwhal": "Narwhal", "Beluga_WhiteWhale": "Beluga",
        "BeardedSeal": "Bearded seal", "RingedSeal": "Ringed seal", "Walrus": "Walrus"}
LATIN = {"HumpbackWhale": "Megaptera novaeangliae", "BowheadWhale": "Balaena mysticetus",
         "NorthernRightWhale": "Eubalaena glacialis", "KillerWhale": "Orcinus orca", "Narwhal": "Monodon monoceros",
         "Beluga_WhiteWhale": "Delphinapterus leucas", "BeardedSeal": "Erignathus barbatus",
         "RingedSeal": "Pusa hispida", "Walrus": "Odobenus rosmarus"}
ORDER = ["HumpbackWhale", "BowheadWhale", "NorthernRightWhale", "KillerWhale", "Narwhal",
         "Beluga_WhiteWhale", "BeardedSeal", "RingedSeal", "Walrus"]
SPECIES_GROUP = {s: g for g, ss in [("Mysticete", ORDER[:3]), ("Odontocete", ORDER[3:6]), ("Pinniped", ORDER[6:])] for s in ss}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

def save(name):
    plt.savefig(f"{FIG}/{name}.png"); plt.close(); print("wrote", name)

def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

# ------------------------------------------------------------------ data (same order and filters as analysis.py)
COLS = ([f"mfccMean{i}" for i in range(20)] + [f"mfccStd{i}" for i in range(20)] +
        [f"deltaMfccMean{i}" for i in range(20)] + [f"deltaMfccStd{i}" for i in range(20)] +
        [f"{n}{s}" for n in ["centroid", "bandwidth", "rolloff", "zcr", "rms"] for s in ["Mean", "Std"]])
Xs, Ms, metas = [], [], []
for f in sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz"))):
    d = np.load(f, allow_pickle=True)
    Xs.append(d["X"]); Ms.append(d["mel"]); metas.append(pd.DataFrame(d["meta"].tolist(),
        columns=["species", "year", "filename", "durationSeconds", "originalSr"]))
X = pd.DataFrame(np.vstack(Xs), columns=COLS); mel = np.concatenate(Ms)
meta = pd.concat(metas, ignore_index=True)
meta["group"] = meta["species"].map(SPECIES_GROUP); meta["session"] = meta["species"] + "_" + meta["filename"].str[:5]
keep = np.isfinite(X.values).all(axis=1)
X, mel, meta = X[keep].reset_index(drop=True), mel[keep], meta[keep].reset_index(drop=True)
summ = pd.read_csv(f"{OUT}/model_summary.csv")
N = len(meta)

# ------------------------------------------------------------------ Fig 1: dataset overview
cnt = meta.groupby("species").agg(clips=("filename", "size"), sessions=("session", "nunique")).reindex(ORDER)[::-1]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, col, lab in [(axs[0], "clips", "Clips"), (axs[1], "sessions", "Recording sessions (tapes)")]:
    ax.barh(range(9), cnt[col], color=[GROUP_C[SPECIES_GROUP[s]] for s in cnt.index], height=0.7, edgecolor=SURF, lw=1)
    for i, v in enumerate(cnt[col]): ax.text(v, i, f" {v}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel(lab); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, cnt[col].max() * 1.15)
axs[0].set_yticks(range(9)); axs[0].set_yticklabels([NICE[s] for s in cnt.index])
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_C[g]) for g in GROUP_C]
fig.legend(handles, GROUP_NAME.values(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.06))
fig.suptitle(f"Dataset: {N:,} Watkins clips from {meta['session'].nunique()} recording sessions",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
plt.tight_layout(rect=(0, 0.04, 1, 1)); save("paper_01_dataset_overview")

# ------------------------------------------------------------------ Fig 2: example spectrograms
fig, axs = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True)
for ax, s in zip(axs.ravel(), ORDER):
    idx = np.where(meta["species"].values == s)[0]
    if len(idx) == 0: ax.set_axis_off(); continue
    full = idx[meta["durationSeconds"].values[idx].astype(float) >= 1.5]
    idx = full if len(full) else idx
    energy = mel[idx].astype(float).mean(axis=(1, 2)); i = idx[np.argsort(energy)[len(idx) // 2]]
    img = mel[i].astype(float); lo, hi = np.percentile(img, [30, 99.5])
    ax.imshow(img, origin="lower", aspect="auto", cmap="Greys", vmin=lo, vmax=hi)
    ax.set_title(f"{NICE[s]}", fontsize=9, color=GROUP_C[SPECIES_GROUP[s]])
    ax.text(0.98, 0.95, LATIN[s], transform=ax.transAxes, ha="right", va="top", fontsize=7, style="italic",
            color=INK2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs[-1]: ax.set_xlabel("Time (128 frames)")
for ax in axs[:, 0]: ax.set_ylabel("Mel band (64)")
fig.suptitle("Example log mel spectrograms, one typical clip per species", x=0.02, ha="left", fontweight="bold", fontsize=11)
plt.tight_layout(); save("paper_02_example_spectrograms")

# ------------------------------------------------------------------ Fig 3: leakage, grouped vs random split
rf = summ[summ["model"] == "random forest"]
tasks = ["Species (9 way)", "Taxonomic group (3 way)"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
for ax, metric, lab in [(axs[0], "accuracy", "Accuracy"), (axs[1], "balancedAccuracy", "Balanced accuracy")]:
    for j, (cv, c, nm) in enumerate([("5 fold, random", GREY, "Random split (clips from one tape on both sides)"),
                                      ("5 fold, grouped", C1, "Grouped split (whole tapes held out)")]):
        vals = [rf[(rf.task == t) & (rf.cv == cv)][metric].iloc[0] for t in tasks]
        sds = [rf[(rf.task == t) & (rf.cv == cv)][metric + "Sd"].iloc[0] for t in tasks]
        x = np.arange(2) + (j - 0.5) * 0.36
        ax.bar(x, vals, 0.34, color=c, yerr=sds, capsize=3, error_kw=dict(ecolor=INK2, lw=0.8), label=nm, edgecolor=SURF, lw=1)
        for xi, v in zip(x, vals): ax.text(xi, 0.03, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(2)); ax.set_xticklabels(["9 species", "3 groups"]); ax.set_ylim(0, 1.05)
    ax.set_title(lab); ygrid(ax); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Random splits inflate random forest scores by leaking recording conditions",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
plt.tight_layout(rect=(0, 0.05, 1, 1)); save("paper_03_leakage_grouped_vs_random")

# ------------------------------------------------------------------ Fig 4: model comparison (grouped CV only)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
models = [("majority class baseline", GREY, "Majority class baseline"), ("random forest", C1, "Random forest, 90 acoustic features"),
          ("transformer (from scratch, log mel)", C2, "Transformer, log mel spectrogram")]
for ax, t in zip(axs, tasks):
    for j, (m, c, nm) in enumerate(models):
        r = summ[(summ.task == t) & (summ.model == m) & ((summ.cv == "5 fold, grouped") | (summ.cv.isna()) | (summ.cv == ""))]
        if r.empty: continue
        r = r.iloc[0]
        for k, metric in enumerate(["accuracy", "balancedAccuracy"]):
            x = k + (j - 1) * 0.27; v = r[metric]; sd = r[metric + "Sd"] if not np.isnan(r[metric + "Sd"]) else 0
            ax.bar(x, v, 0.25, color=c, yerr=sd if sd else None, capsize=3, error_kw=dict(ecolor=INK2, lw=0.8),
                   label=nm if k == 0 else None, edgecolor=SURF, lw=1)
            ax.text(x, v + sd + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Accuracy", "Balanced accuracy"]); ax.set_ylim(0, 1.1)
    ax.set_title("9 species" if "Species" in t else "3 taxonomic groups"); ygrid(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
h, l = [], []
for a in axs:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Model comparison, 5 fold cross validation grouped by recording session", x=0.02, ha="left", fontweight="bold", fontsize=11)
plt.tight_layout(rect=(0, 0.06, 1, 1)); save("paper_04_model_comparison")

# ------------------------------------------------------------------ RF out of fold predictions (same folds as analysis.py)
groups = meta["session"].values
def oof(y, grouped):
    y = np.asarray(y)
    sp = (StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    p = np.empty(len(y), dtype=object)
    for tr, te in sp:
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=2, random_state=0)
        clf.fit(X.iloc[tr], y[tr]); p[te] = clf.predict(X.iloc[te])
    return p
ys = meta["species"].values
p_g, p_r = oof(ys, True), oof(ys, False)
f1g = f1_score(ys, p_g, labels=ORDER, average=None); f1r = f1_score(ys, p_r, labels=ORDER, average=None)
pd.DataFrame({"species": ORDER, "f1Grouped": f1g, "f1Random": f1r, "clips": cnt.reindex(ORDER)["clips"].values,
              "sessions": cnt.reindex(ORDER)["sessions"].values}).to_csv(f"{OUT}/rf_per_species_f1.csv", index=False)

# ------------------------------------------------------------------ Fig 5: per species F1, dumbbell
fig, ax = plt.subplots(figsize=(7, 4))
yy = np.arange(9)[::-1]
for y_, a, b in zip(yy, f1g, f1r): ax.plot([a, b], [y_, y_], color=GRID, lw=3, zorder=1)
ax.scatter(f1r, yy, s=55, color=GREY, zorder=2, label="Random split", edgecolor=SURF, lw=1.5)
ax.scatter(f1g, yy, s=55, color=C1, zorder=3, label="Grouped split", edgecolor=SURF, lw=1.5)
for y_, a, b in zip(yy, f1g, f1r): ax.text(min(a, b) - 0.03, y_, f"{a:.2f}", ha="right", va="center", fontsize=7, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels([f"{NICE[s]} ({int(c) if c == c else 0} tapes)" for s, c in zip(ORDER, cnt.reindex(ORDER)["sessions"])])
ax.set_xlim(-0.08, 1.02); ax.set_xlabel("F1 score (random forest, out of fold)"); ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("Per species F1, grouped vs random split")
plt.tight_layout(); save("paper_05_per_species_f1")

# ------------------------------------------------------------------ Fig 6: normalized confusion matrix, RF grouped species
cm = confusion_matrix(ys, p_g, labels=ORDER); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
for i in range(9):
    for j in range(9):
        if cm[i, j]: ax.text(j, i, f"{cmn[i, j]:.0%}".replace("%", ""), ha="center", va="center", fontsize=7,
                             color="white" if cmn[i, j] > 0.55 else INK)
ax.set_xticks(range(9)); ax.set_xticklabels([NICE[s] for s in ORDER], rotation=45, ha="right")
ax.set_yticks(range(9)); ax.set_yticklabels([NICE[s] for s in ORDER])
for k in (2.5, 5.5): ax.axhline(k, color=INK2, lw=0.8); ax.axvline(k, color=INK2, lw=0.8)
ax.spines[:].set_visible(False); ax.set_xlabel("Predicted species"); ax.set_ylabel("True species")
cb = plt.colorbar(im, fraction=0.046, pad=0.04); cb.set_label("Row percentage"); cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.set_title("Random forest confusion matrix, grouped CV (percent of true class)")
plt.tight_layout(); save("paper_06_confusion_species_rf")

# ------------------------------------------------------------------ Fig 7: feature importance by feature family
fi = pd.read_csv(f"{OUT}/05_species_rf_feature_importance.csv", index_col=0).iloc[:, 0]
def family(n):
    for k, v in [("deltaMfccMean", "Delta MFCC mean"), ("deltaMfccStd", "Delta MFCC variability"), ("mfccMean", "MFCC mean"),
                 ("mfccStd", "MFCC variability"), ("centroid", "Spectral centroid"), ("bandwidth", "Spectral bandwidth"),
                 ("rolloff", "Spectral rolloff"), ("zcr", "Zero crossing rate"), ("rms", "Loudness (RMS)")]:
        if n.startswith(k): return v
top = fi.sort_values(ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.barh(range(len(top)), top.values, color=C1, height=0.7, edgecolor=SURF, lw=1)
ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{n}  ({family(n)})" for n in top.index], fontsize=8)
ax.set_xlabel("Mean decrease in impurity"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Top 15 acoustic features, species random forest")
plt.tight_layout(); save("paper_07_feature_importance")

# ------------------------------------------------------------------ Fig 8: PCA by group
Xp = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(X))
ev = Xp.explained_variance_ratio_; Z = Xp.transform(StandardScaler().fit_transform(X))
fig, ax = plt.subplots(figsize=(6.4, 4.6))
rng = np.random.default_rng(0); order = rng.permutation(N)
ax.scatter(Z[order, 0], Z[order, 1], s=9, c=[GROUP_C[g] for g in meta["group"].values[order]], alpha=0.6,
           edgecolor=SURF, lw=0.3)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GROUP_C[g], label=GROUP_NAME[g]) for g in GROUP_C],
          frameon=False, loc="upper right")
ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
ax.set_title("Acoustic feature space overlaps heavily between groups (PCA)")
plt.tight_layout(); save("paper_08_pca_by_group")
print("done")
