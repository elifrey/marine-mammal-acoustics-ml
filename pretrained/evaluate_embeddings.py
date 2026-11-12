"""Linear probe (and random forest) on pretrained embeddings, 5 fold CV grouped by session and random.

Usage: python pretrained/evaluate_embeddings.py birdnet|perch2 lr|rf species|group [grouped,random]
Uses outputs/clip_metadata.csv (written by analysis.py) for labels and sessions, so folds match the main analysis.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
warnings.filterwarnings('ignore')

emb_name, which, task = sys.argv[1], sys.argv[2], sys.argv[3]
schemes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['grouped', 'random']
meta = pd.read_csv('outputs/clip_metadata.csv')
e = np.load(f'feat/{emb_name}_embeddings.npz')
key = {n.split('/')[0] + '/' + n.split('/')[-1]: i for i, n in enumerate(e['names'])}
X = e['emb'][[key[s + '/' + f] for s, f in zip(meta.species, meta.filename)]].astype(np.float64)
y, groups = meta[task].values, meta.session.values

rows, perclass = [], {}
for cv in schemes:
    split = (StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups) if cv == 'grouped'
             else StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    A, B, F = [], [], []; pred = np.empty(len(y), dtype=object)
    for tr, te in split:
        clf = (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced'))
               if which == 'lr' else RandomForestClassifier(n_estimators=400, class_weight='balanced', n_jobs=-1, random_state=0))
        clf.fit(X[tr], y[tr]); p = clf.predict(X[te]); pred[te] = p
        A.append(accuracy_score(y[te], p)); B.append(balanced_accuracy_score(y[te], p)); F.append(f1_score(y[te], p, average='macro'))
    rows.append(dict(model=f'{emb_name}+{which}', task=task, cv=cv, accuracy=np.mean(A), accuracySd=np.std(A),
                     balancedAccuracy=np.mean(B), balancedAccuracySd=np.std(B), macroF1=np.mean(F)))
    labs = sorted(set(y)); perclass[cv] = dict(zip(labs, np.round(f1_score(y, pred, labels=labs, average=None), 3).tolist()))
os.makedirs('outputs', exist_ok=True)
json.dump(dict(rows=rows, perclass=perclass), open(f'outputs/{emb_name}_{which}_{task}_{"_".join(schemes)}.json', 'w'), indent=1)
for r in rows: print(r)
