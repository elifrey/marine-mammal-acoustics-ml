"""Linear probe (and random forest) on pretrained embeddings, 5 fold CV grouped by session and random.

Usage: python pretrained/evaluate_embeddings.py birdnet|perch2 lr|rf species|group [grouped,random]
Uses outputs/clip_metadata.csv (written by analysis.py) for labels and sessions, so folds match the main analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# run as `python pretrained/evaluate_embeddings.py` from the repository root,
# so the shared module one level up is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import OUT_DIR, RF_TREES, SEED, make_folds  # noqa: E402

warnings.filterwarnings("ignore")  # convergence warnings from the linear probe


def load_embeddings(emb_name: str, meta: pd.DataFrame) -> np.ndarray:
    """Embedding rows reordered to match outputs/clip_metadata.csv."""
    data = np.load(f"feat/{emb_name}_embeddings.npz")
    row_of = {n.split("/")[0] + "/" + n.split("/")[-1]: i for i, n in enumerate(data["names"])}
    rows = [row_of[s + "/" + f] for s, f in zip(meta.species, meta.filename)]
    return data["emb"][rows].astype(np.float64)


def make_classifier(which: str):
    """'lr': standardised logistic regression (linear probe); 'rf': random forest."""
    if which == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    return RandomForestClassifier(n_estimators=RF_TREES, class_weight="balanced", n_jobs=-1,
                                  random_state=SEED)


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, scheme: str,
             which: str) -> tuple[dict, np.ndarray]:
    """Mean and SD of the fold scores, plus out of fold predictions for per class F1."""
    accs, baccs, f1s = [], [], []
    predictions = np.empty(len(y), dtype=object)
    for train, test in make_folds(X, y, groups, grouped=scheme == "grouped"):
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


def main(emb_name: str, which: str, task: str, schemes: list[str]) -> None:
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

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f'{OUT_DIR}/{emb_name}_{which}_{task}_{"_".join(schemes)}.json'
    with open(out_path, "w") as fh:
        json.dump(dict(rows=rows, perclass=per_class), fh, indent=1)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify pretrained embeddings with grouped and random CV.")
    parser.add_argument("embedding", choices=["birdnet", "perch2"])
    parser.add_argument("classifier", choices=["lr", "rf"])
    parser.add_argument("task", choices=["species", "group"])
    parser.add_argument("schemes", nargs="?", default="grouped,random", help="comma separated: grouped,random")
    args = parser.parse_args()
    main(args.embedding, args.classifier, args.task, args.schemes.split(","))
