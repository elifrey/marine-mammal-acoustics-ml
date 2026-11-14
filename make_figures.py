"""
Publication figures. Run after analysis.py:

    python make_figures.py [feat_dir]

Reads outputs/model_summary.csv and outputs/clip_metadata.csv, recomputes the
random forest out of fold predictions (same folds and seeds as analysis.py)
for per species scores, and writes figures/paper_*.png (300 dpi) plus
outputs/rf_per_species_f1.csv.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from common import SEED, SPECIES_GROUP, load_features, make_folds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

FIG_DIR, OUT_DIR = "figures", "outputs"

# ---------------------------------------------------------------- style
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
GROUP_COLOUR = {"Mysticete": C1, "Odontocete": C2, "Pinniped": C3}
GROUP_NAME = {"Mysticete": "Baleen whales", "Odontocete": "Toothed whales", "Pinniped": "Seals and walrus"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

ORDER = ["HumpbackWhale", "BowheadWhale", "NorthernRightWhale", "KillerWhale", "Narwhal",
         "Beluga_WhiteWhale", "BeardedSeal", "RingedSeal", "Walrus"]
COMMON_NAME = {
    "HumpbackWhale": "Humpback whale", "BowheadWhale": "Bowhead whale",
    "NorthernRightWhale": "N. right whale", "KillerWhale": "Killer whale", "Narwhal": "Narwhal",
    "Beluga_WhiteWhale": "Beluga", "BeardedSeal": "Bearded seal", "RingedSeal": "Ringed seal",
    "Walrus": "Walrus",
}
LATIN_NAME = {
    "HumpbackWhale": "Megaptera novaeangliae", "BowheadWhale": "Balaena mysticetus",
    "NorthernRightWhale": "Eubalaena glacialis", "KillerWhale": "Orcinus orca",
    "Narwhal": "Monodon monoceros", "Beluga_WhiteWhale": "Delphinapterus leucas",
    "BeardedSeal": "Erignathus barbatus", "RingedSeal": "Pusa hispida", "Walrus": "Odobenus rosmarus",
}
TASKS = ["Species (9 way)", "Taxonomic group (3 way)"]
FEATURE_FAMILIES = [
    ("deltaMfccMean", "Delta MFCC mean"), ("deltaMfccStd", "Delta MFCC variability"),
    ("mfccMean", "MFCC mean"), ("mfccStd", "MFCC variability"), ("centroid", "Spectral centroid"),
    ("bandwidth", "Spectral bandwidth"), ("rolloff", "Spectral rolloff"),
    ("zcr", "Zero crossing rate"), ("rms", "Loudness (RMS)"),
]


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(name):
    plt.savefig(f"{FIG_DIR}/{name}.png")
    plt.close()
    print("wrote", name)


def add_ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def suptitle(fig, text):
    fig.suptitle(text, x=0.02, ha="left", fontweight="bold", fontsize=11)


# ---------------------------------------------------------------- figures

def fig_dataset_overview(meta, counts):
    """Fig 1: clips and recording sessions per species."""
    counts = counts[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, column, label in [(axs[0], "clips", "Clips"), (axs[1], "sessions", "Recording sessions (tapes)")]:
        colours = [GROUP_COLOUR[SPECIES_GROUP[s]] for s in counts.index]
        ax.barh(range(9), counts[column], color=colours, height=0.7, edgecolor=SURF, lw=1)
        for i, value in enumerate(counts[column]):
            ax.text(value, i, f" {value}", va="center", fontsize=8, color=INK2)
        ax.set_xlabel(label)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, counts[column].max() * 1.15)
    axs[0].set_yticks(range(9))
    axs[0].set_yticklabels([COMMON_NAME[s] for s in counts.index])
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOUR[g]) for g in GROUP_COLOUR]
    fig.legend(handles, GROUP_NAME.values(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.55, -0.06))
    suptitle(fig, f"Dataset: {len(meta):,} Watkins clips from {meta['session'].nunique()} recording sessions")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save("paper_01_dataset_overview")


def typical_clip(meta, mel, species):
    """Index of the clip with median energy among clips of at least 1.5 s (if any)."""
    idx = np.where(meta["species"].values == species)[0]
    if len(idx) == 0:
        return None
    full_length = idx[meta["durationSeconds"].values[idx].astype(float) >= 1.5]
    idx = full_length if len(full_length) else idx
    energy = mel[idx].astype(float).mean(axis=(1, 2))
    return idx[np.argsort(energy)[len(idx) // 2]]


def fig_example_spectrograms(meta, mel):
    """Fig 2: one typical log mel spectrogram per species."""
    fig, axs = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True)
    for ax, species in zip(axs.ravel(), ORDER):
        i = typical_clip(meta, mel, species)
        if i is None:
            ax.set_axis_off()
            continue
        img = mel[i].astype(float)
        lo, hi = np.percentile(img, [30, 99.5])
        ax.imshow(img, origin="lower", aspect="auto", cmap="Greys", vmin=lo, vmax=hi)
        ax.set_title(COMMON_NAME[species], fontsize=9, color=GROUP_COLOUR[SPECIES_GROUP[species]])
        ax.text(0.98, 0.95, LATIN_NAME[species], transform=ax.transAxes, ha="right", va="top",
                fontsize=7, style="italic", color=INK2,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[-1]:
        ax.set_xlabel("Time (128 frames)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Mel band (64)")
    suptitle(fig, "Example log mel spectrograms, one typical clip per species")
    plt.tight_layout()
    save("paper_02_example_spectrograms")


def fig_leakage(summary):
    """Fig 3: random forest scores with grouped versus random splits."""
    rf = summary[summary["model"] == "random forest"]
    splits = [("5 fold, random", GREY, "Random split (clips from one tape on both sides)"),
              ("5 fold, grouped", C1, "Grouped split (whole tapes held out)")]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, metric, title in [(axs[0], "accuracy", "Accuracy"), (axs[1], "balancedAccuracy", "Balanced accuracy")]:
        for j, (cv, colour, label) in enumerate(splits):
            rows = [rf[(rf.task == t) & (rf.cv == cv)].iloc[0] for t in TASKS]
            values = [r[metric] for r in rows]
            sds = [r[metric + "Sd"] for r in rows]
            x = np.arange(2) + (j - 0.5) * 0.36
            ax.bar(x, values, 0.34, color=colour, yerr=sds, capsize=3, error_kw=dict(ecolor=INK2, lw=0.8),
                   label=label, edgecolor=SURF, lw=1)
            for xi, value in zip(x, values):
                ax.text(xi, 0.03, f"{value:.0%}", ha="center", va="bottom", fontsize=8,
                        color="white", fontweight="bold")
        ax.set_xticks(range(2))
        ax.set_xticklabels(["9 species", "3 groups"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        add_ygrid(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    suptitle(fig, "Random splits inflate random forest scores by leaking recording conditions")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    save("paper_03_leakage_grouped_vs_random")


def fig_model_comparison(summary):
    """Fig 4: baseline, random forest and transformer, grouped CV only."""
    models = [("majority class baseline", GREY, "Majority class baseline"),
              ("random forest", C1, "Random forest, 90 acoustic features"),
              ("transformer (from scratch, log mel)", C2, "Transformer, log mel spectrogram")]
    grouped_or_none = (summary.cv == "5 fold, grouped") | summary.cv.isna() | (summary.cv == "")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, task in zip(axs, TASKS):
        for j, (model, colour, label) in enumerate(models):
            rows = summary[(summary.task == task) & (summary.model == model) & grouped_or_none]
            if rows.empty:
                continue
            row = rows.iloc[0]
            for k, metric in enumerate(["accuracy", "balancedAccuracy"]):
                x = k + (j - 1) * 0.27
                value = row[metric]
                sd = 0 if np.isnan(row[metric + "Sd"]) else row[metric + "Sd"]
                ax.bar(x, value, 0.25, color=colour, yerr=sd if sd else None, capsize=3,
                       error_kw=dict(ecolor=INK2, lw=0.8), label=label if k == 0 else None,
                       edgecolor=SURF, lw=1)
                ax.text(x, value + sd + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Accuracy", "Balanced accuracy"])
        ax.set_ylim(0, 1.1)
        ax.set_title("9 species" if "Species" in task else "3 taxonomic groups")
        add_ygrid(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    suptitle(fig, "Model comparison, 5 fold cross validation grouped by recording session")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    save("paper_04_model_comparison")


def out_of_fold_predictions(features, y, groups, grouped):
    """Random forest predictions for every clip from the fold where it was held out."""
    y = np.asarray(y)
    predictions = np.empty(len(y), dtype=object)
    for train, test in make_folds(features, y, groups, grouped):
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=2, random_state=SEED)
        clf.fit(features.iloc[train], y[train])
        predictions[test] = clf.predict(features.iloc[test])
    return predictions


def fig_per_species_f1(f1_grouped, f1_random, sessions):
    """Fig 5: per species F1 for grouped and random splits (dumbbell plot)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = np.arange(9)[::-1]
    for y, a, b in zip(rows, f1_grouped, f1_random):
        ax.plot([a, b], [y, y], color=GRID, lw=3, zorder=1)
    ax.scatter(f1_random, rows, s=55, color=GREY, zorder=2, label="Random split", edgecolor=SURF, lw=1.5)
    ax.scatter(f1_grouped, rows, s=55, color=C1, zorder=3, label="Grouped split", edgecolor=SURF, lw=1.5)
    for y, a, b in zip(rows, f1_grouped, f1_random):
        ax.text(min(a, b) - 0.03, y, f"{a:.2f}", ha="right", va="center", fontsize=7, color=INK2)
    ax.set_yticks(rows)
    ax.set_yticklabels([f"{COMMON_NAME[s]} ({int(n) if n == n else 0} tapes)" for s, n in zip(ORDER, sessions)])
    ax.set_xlim(-0.08, 1.02)
    ax.set_xlabel("F1 score (random forest, out of fold)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title("Per species F1, grouped vs random split")
    plt.tight_layout()
    save("paper_05_per_species_f1")


def fig_confusion(y, predictions):
    """Fig 6: normalised confusion matrix, species random forest, grouped CV."""
    cm = confusion_matrix(y, predictions, labels=ORDER)
    row_share = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(row_share, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, f"{row_share[i, j]:.0%}".replace("%", ""), ha="center", va="center", fontsize=7,
                        color="white" if row_share[i, j] > 0.55 else INK)
    ax.set_xticks(range(9))
    ax.set_xticklabels([COMMON_NAME[s] for s in ORDER], rotation=45, ha="right")
    ax.set_yticks(range(9))
    ax.set_yticklabels([COMMON_NAME[s] for s in ORDER])
    for boundary in (2.5, 5.5):  # lines between baleen whales, toothed whales and pinnipeds
        ax.axhline(boundary, color=INK2, lw=0.8)
        ax.axvline(boundary, color=INK2, lw=0.8)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    colourbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    colourbar.set_label("Row percentage")
    colourbar.outline.set_visible(False)
    colourbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Random forest confusion matrix, grouped CV (percent of true class)")
    plt.tight_layout()
    save("paper_06_confusion_species_rf")


def feature_family(name):
    for prefix, family in FEATURE_FAMILIES:
        if name.startswith(prefix):
            return family
    return None


def fig_feature_importance():
    """Fig 7: top 15 random forest features, labelled by feature family."""
    importance = pd.read_csv(f"{OUT_DIR}/05_species_rf_feature_importance.csv", index_col=0).iloc[:, 0]
    top = importance.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(range(len(top)), top.values, color=C1, height=0.7, edgecolor=SURF, lw=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{n}  ({feature_family(n)})" for n in top.index], fontsize=8)
    ax.set_xlabel("Mean decrease in impurity")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Top 15 acoustic features, species random forest")
    plt.tight_layout()
    save("paper_07_feature_importance")


def fig_pca(features, meta):
    """Fig 8: first two principal components of the acoustic features, by group."""
    pca = PCA(n_components=2, random_state=SEED).fit(StandardScaler().fit_transform(features))
    explained = pca.explained_variance_ratio_
    scores = pca.transform(StandardScaler().fit_transform(features))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    order = np.random.default_rng(SEED).permutation(len(meta))  # random draw order avoids one group on top
    ax.scatter(scores[order, 0], scores[order, 1], s=9,
               c=[GROUP_COLOUR[g] for g in meta["group"].values[order]], alpha=0.6, edgecolor=SURF, lw=0.3)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GROUP_COLOUR[g], label=GROUP_NAME[g])
                       for g in GROUP_COLOUR], frameon=False, loc="upper right")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
    ax.set_title("Acoustic feature space overlaps heavily between groups (PCA)")
    plt.tight_layout()
    save("paper_08_pca_by_group")


# ---------------------------------------------------------------- main

def main(feat_dir="feat"):
    set_style()
    features, mel, meta = load_features(feat_dir)
    summary = pd.read_csv(f"{OUT_DIR}/model_summary.csv")
    counts = (meta.groupby("species")
              .agg(clips=("filename", "size"), sessions=("session", "nunique"))
              .reindex(ORDER))

    fig_dataset_overview(meta, counts)
    fig_example_spectrograms(meta, mel)
    fig_leakage(summary)
    fig_model_comparison(summary)

    species = meta["species"].values
    groups = meta["session"].values
    pred_grouped = out_of_fold_predictions(features, species, groups, grouped=True)
    pred_random = out_of_fold_predictions(features, species, groups, grouped=False)
    f1_grouped = f1_score(species, pred_grouped, labels=ORDER, average=None)
    f1_random = f1_score(species, pred_random, labels=ORDER, average=None)
    pd.DataFrame({"species": ORDER, "f1Grouped": f1_grouped, "f1Random": f1_random,
                  "clips": counts["clips"].values, "sessions": counts["sessions"].values}
                 ).to_csv(f"{OUT_DIR}/rf_per_species_f1.csv", index=False)

    fig_per_species_f1(f1_grouped, f1_random, counts["sessions"])
    fig_confusion(species, pred_grouped)
    fig_feature_importance()
    fig_pca(features, meta)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "feat")
