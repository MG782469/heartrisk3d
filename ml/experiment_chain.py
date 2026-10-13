"""Experiment: does a classifier chain (vessel model also sees the predicted CAD / LAD probability)
improve LCX and RCA? Training split only, repeated 5x5 CV with the chain fitted inside each fold
(no leakage). The test set is not touched. Result decides whether the chain is deployed."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import features as F  # noqa: E402
from ml.train import make_model, raw_score  # noqa: E402

raw = pd.read_csv(ROOT / "data" / "zalizadeh_sani_ext.csv")
X_all, Y_all = F.encode(raw), F.labels(raw)
tr, _ = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=F.RANDOM_SEED, stratify=F.nvessels(Y_all))
X = X_all.iloc[tr][F.FEATURE_NAMES].to_numpy()
Y = Y_all.iloc[tr]
C = {"CAD": 0.1, "LAD": 0.03, "RCA": 0.03}
FAM = {"CAD": "logreg", "LAD": "logreg", "LCX": "gbm", "RCA": "logreg"}

results = {}
for tgt in ["LCX", "RCA"]:
    y = Y[tgt].to_numpy()
    base, chain = [], []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=F.RANDOM_SEED)
    for a, b in rskf.split(X, y):
        m = make_model(FAM[tgt], C=C.get(tgt)).fit(X[a], y[a])
        base.append(roc_auc_score(y[b], raw_score(m, X[b])))
        # chain inputs: CAD and LAD scores, out-of-fold on the outer-train part, refit for outer-val
        extra_tr, extra_va = [], []
        for up in ["CAD", "LAD"]:
            yu = Y[up].to_numpy()
            inner = StratifiedKFold(5, shuffle=True, random_state=F.RANDOM_SEED)
            oof = cross_val_predict(make_model("logreg", C=C[up]), X[a], yu[a], cv=inner, method="decision_function")
            extra_tr.append(oof)
            extra_va.append(raw_score(make_model("logreg", C=C[up]).fit(X[a], yu[a]), X[b]))
        Xa = np.column_stack([X[a]] + extra_tr)
        Xb = np.column_stack([X[b]] + extra_va)
        mc = make_model(FAM[tgt], C=C.get(tgt)).fit(Xa, y[a])
        chain.append(roc_auc_score(y[b], raw_score(mc, Xb)))
    base, chain = np.array(base), np.array(chain)
    diff = chain - base
    results[tgt] = {"base": round(base.mean(), 4), "chain": round(chain.mean(), 4),
                    "gain": round(diff.mean(), 4), "gain_se": round(diff.std() / np.sqrt(len(diff)), 4)}
    print(tgt, results[tgt], flush=True)
(ROOT / "artifacts" / "reports" / "chain_experiment.json").write_text(json.dumps(results, indent=1))
