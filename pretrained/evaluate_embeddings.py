"""Linear probe (and random forest) on pretrained embeddings, 5 fold CV grouped by session and random.

Usage: python pretrained/evaluate_embeddings.py birdnet|perch2 lr|rf species|group [grouped,random]
Uses outputs/clip_metadata.csv (written by analysis.py) for labels and sessions, so folds match the main analysis.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_embeddings(emb_name, meta):
    """Embedding rows reordered to match outputs/clip_metadata.csv."""
    data = np.load(f"feat/{emb_name}_embeddings.npz")
    row_of = {n.split("/")[0] + "/" + n.split("/")[-1]: i for i, n in enumerate(data["names"])}
    rows = [row_of[s + "/" + f] for s, f in zip(meta.species, meta.filename)]
    return data["emb"][rows].astype(np.float64)


def make_classifier(which):
    if which == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    return RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0)


def folds(X, y, groups, scheme):
    if scheme == "grouped":
        return StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups)
    return StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)


def evaluate(X, y, groups, scheme, which):
    """Mean and SD of the fold scores, plus out of fold predictions for per class F1."""
    accs, baccs, f1s = [], [], []
    predictions = np.empty(len(y), dtype=object)
    for train, test in folds(X, y, groups, scheme):
        clf = make_classifier(which)
        clf.fit(X[train], y[train])
        pred = clf.predict(X[test])
        predictions[test] = pred
        accs.append(accuracy_score(y[test], pred))
        baccs.append(balanced_accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average="macro"))
    scores = dict(accuracy=np.mean(accs), accuracySd=np.std(accs),
                  balancedAccuracy=np.mean(baccs), balancedAccuracySd=np.std(baccs), macroF1=np.mean(f1s))
    return scores, predictions


def main(emb_name, which, task, schemes):
    meta = pd.read_csv("outputs/clip_metadata.csv")
    X = load_embeddings(emb_name, meta)
    y, groups = meta[task].values, meta.session.values
    labels = sorted(set(y))

    rows, per_class = [], {}
    for scheme in schemes:
        scores, predictions = evaluate(X, y, groups, scheme, which)
        rows.append(dict(model=f"{emb_name}+{which}", task=task, cv=scheme, **scores))
        f1 = f1_score(y, predictions, labels=labels, average=None)
        per_class[scheme] = dict(zip(labels, np.round(f1, 3).tolist()))

    os.makedirs("outputs", exist_ok=True)
    out_path = f'outputs/{emb_name}_{which}_{task}_{"_".join(schemes)}.json'
    with open(out_path, "w") as fh:
        json.dump(dict(rows=rows, perclass=per_class), fh, indent=1)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4].split(",") if len(sys.argv) > 4 else ["grouped", "random"])
