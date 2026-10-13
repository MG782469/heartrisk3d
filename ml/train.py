"""Train, validate, calibrate and export the HeartRisk 3D models.

Run:  python ml/train.py

Pipeline
  1. load + encode (targets never used as inputs)
  2. stratified 80/20 split (stratified on number of diseased vessels)
  3. benchmark 4 model families per target with repeated 5x5 CV (nested C tuning for LR)
  4. pick a deployable family per target with the 1-SE rule (simpler model wins ties)
  5. for every modality set x target: out-of-fold predictions -> Platt calibration ->
     Mondrian (class-conditional) conformal thresholds; bootstrap refits for an uncertainty band
  6. held-out test metrics with bootstrap 95% CI, conformal coverage, reports
  7. export: artifacts/models.joblib (sklearn) + artifacts/model_bundle.json (browser engine)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import features as F  # noqa: E402

ART = ROOT / "artifacts"
REP = ART / "reports"
ALPHA = 0.10           # conformal miscoverage -> 90% prediction sets
N_SPLITS, N_REPEATS = 5, 5
N_BOOT_MODELS = 10     # bootstrap refits for the probability band
N_BOOT_CI = 1000       # bootstrap resamples for test-metric confidence intervals
SEED = F.RANDOM_SEED
TARGETS = list(F.TARGETS)
DEPLOYABLE = ["logreg", "gbm"]  # families the browser engine can run exactly


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- models
def make_model(family, C=None, monotone=None):
    if family == "logreg":
        if C is None:
            clf = LogisticRegressionCV(Cs=[0.03, 0.1, 0.3, 1.0], penalty="elasticnet", solver="saga",
                                       l1_ratios=[0.5], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                       scoring="roc_auc", class_weight="balanced", max_iter=5000,
                                       random_state=SEED)
        else:
            clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=C,
                                     class_weight="balanced", max_iter=5000, random_state=SEED)
    elif family == "gbm":
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                         random_state=SEED)
    elif family == "rf":
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=-1)
    elif family == "hgb_monotone":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             monotonic_cst=monotone, random_state=SEED)
    else:
        raise ValueError(family)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def raw_score(model, X):
    """Log-odds style score before calibration."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        return model.decision_function(X)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def monotone_for(cols):
    up = {"Age", "DM", "HTN", "Typical Chest Pain"}
    return [1 if c in up else 0 for c in cols]


def oof_scores(family, X, y, C=None, cols=None):
    """Repeated stratified CV. Returns mean OOF raw score per sample and per-fold AUCs."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    aucs = []
    for tr, va in rskf.split(X, y):
        m = make_model(family, C=C, monotone=monotone_for(cols) if family == "hgb_monotone" else None)
        m.fit(X[tr], y[tr])
        s = raw_score(m, X[va])
        acc[va] += s
        cnt[va] += 1
        aucs.append(roc_auc_score(y[va], s))
    return acc / cnt, np.array(aucs)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(s, y):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(s.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def conformal_thresholds(p, y, alpha=ALPHA):
    """Mondrian (class-conditional) split-conformal on out-of-fold calibrated probabilities.
    Non-conformity score for class c: 1 - P(c). Class c is in the set when 1 - P(c) <= q_c."""
    q = {}
    for c in (0, 1):
        sc = np.sort(1 - (p[y == c] if c == 1 else 1 - p[y == c]))
        n = len(sc)
        k = min(n, math.ceil((n + 1) * (1 - alpha)))
        q[c] = float(sc[k - 1])
    return q


def conformal_set(p, q):
    s1 = ((1 - p) <= q[1]) | (p >= 0.5)
    s0 = (p <= q[0]) | (p < 0.5)  # 1 - P(0) = p
    # the most probable label is always included: the set only grows, so coverage still holds
    return s0, s1


# --------------------------------------------------------------------------- metrics
def point_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, p)
        out["pr_auc"] = average_precision_score(y, p)
    return out


def metrics_with_ci(y, p, rng):
    base = point_metrics(y, p)
    boots = {k: [] for k in base}
    n = len(y)
    for _ in range(N_BOOT_CI):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = point_metrics(y[idx], p[idx])
        for k in boots:
            boots[k].append(m[k])
    return {k: {"value": round(float(base[k]), 4),
                "ci95": [round(float(np.percentile(boots[k], 2.5)), 4),
                         round(float(np.percentile(boots[k], 97.5)), 4)]} for k in base}


def calibration_bins(y, p, n_bins=5):
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = 0, 1
    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p >= lo) & (p <= hi) if hi == 1 else (p >= lo) & (p < hi)
        if m.sum():
            out.append({"pred": round(float(p[m].mean()), 4), "obs": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out


def decision_curve(y, p):
    n = len(y)
    prev = y.mean()
    rows = []
    for t in np.arange(0.05, 0.81, 0.05):
        tp = ((p >= t) & (y == 1)).sum()
        fp = ((p >= t) & (y == 0)).sum()
        nb = tp / n - fp / n * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        rows.append({"threshold": round(float(t), 2), "model": round(float(nb), 4),
                     "treat_all": round(float(nb_all), 4), "treat_none": 0.0})
    return rows


def curve_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    pick = lambda a, k=60: a[np.linspace(0, len(a) - 1, min(k, len(a))).astype(int)]  # noqa: E731
    return {"roc": {"fpr": pick(fpr).round(4).tolist(), "tpr": pick(tpr).round(4).tolist()},
            "pr": {"recall": pick(rec).round(4).tolist(), "precision": pick(prec).round(4).tolist()}}


# --------------------------------------------------------------------------- export helpers
def export_model(model, cols):
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]
    d = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    if isinstance(clf, (LogisticRegression, LogisticRegressionCV)):
        d.update(kind="linear", coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]))
    elif isinstance(clf, GradientBoostingClassifier):
        prior = clf.init_.class_prior_[1]
        trees = []
        for est in clf.estimators_[:, 0]:
            t = est.tree_
            trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(),
                          # full precision: rounding a split threshold can flip a decision at the boundary
                          "f": t.feature.tolist(), "t": [float(v) for v in t.threshold],
                          "v": [float(v) for v in t.value[:, 0, 0]]})
        d.update(kind="gbm", init=float(np.log(prior / (1 - prior))), lr=float(clf.learning_rate), trees=trees)
    else:
        raise TypeError(type(clf))
    return d


def main():
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    REP.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    raw = pd.read_csv(ROOT / "data" / "zalizadeh_sani_ext.csv")
    data_hash = hashlib.sha256((ROOT / "data" / "zalizadeh_sani_ext.csv").read_bytes()).hexdigest()[:12]
    X_all = F.encode(raw)
    Y_all = F.labels(raw)
    assert not set(F.LEAKAGE_COLUMNS) & set(X_all.columns)

    idx_tr, idx_te = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=SEED,
                                      stratify=F.nvessels(Y_all))
    Xtr_df, Xte_df = X_all.iloc[idx_tr], X_all.iloc[idx_te]
    Ytr, Yte = Y_all.iloc[idx_tr], Y_all.iloc[idx_te]
    log(f"train {len(idx_tr)}  test {len(idx_te)}")

    # ---------------------------------------------------------------- 1. benchmark on full feature set
    full_cols = F.FEATURE_NAMES
    Xtr_full = Xtr_df[full_cols].to_numpy()
    benchmark, chosen = {}, {}
    for tgt in TARGETS:
        y = Ytr[tgt].to_numpy()
        res = {}
        for fam in ["logreg", "gbm", "rf", "hgb_monotone"]:
            _, aucs = oof_scores(fam, Xtr_full, y, cols=full_cols)
            res[fam] = {"auc_mean": round(float(aucs.mean()), 4), "auc_std": round(float(aucs.std()), 4),
                        "auc_se": round(float(aucs.std() / math.sqrt(len(aucs))), 4)}
            log(tgt, fam, res[fam])
        best = max(res, key=lambda k: res[k]["auc_mean"])
        floor = res[best]["auc_mean"] - res[best]["auc_se"]
        pick = next((f for f in DEPLOYABLE if res[f]["auc_mean"] >= floor), None)
        rule = "1-SE rule: simplest deployable family within one standard error of the best"
        if pick is None:
            pick = max(DEPLOYABLE, key=lambda k: res[k]["auc_mean"])
            rule = "no deployable family within 1 SE of best; best deployable family used"
        chosen[tgt] = pick
        benchmark[tgt] = {"families": res, "best_overall": best, "chosen": pick, "rule": rule}
        log(tgt, "-> chosen", pick)

    # tune C once per target on the full train set (used by LR refits in every modality set)
    C_of = {}
    for tgt in TARGETS:
        if chosen[tgt] == "logreg":
            m = make_model("logreg")
            m.fit(Xtr_full, Ytr[tgt].to_numpy())
            C_of[tgt] = float(m.named_steps["clf"].C_[0])
    log("C chosen", C_of)

    # ---------------------------------------------------------------- 2. every modality set x target
    sets = F.modality_sets()
    sk_models, bundle_cfg, ablation, test_metrics, conformal_report, oof_store = {}, {}, {}, {}, {}, {}
    for skey, mods in sets.items():
        cols = F.features_for(mods)
        ci = [full_cols.index(c) for c in cols]
        Xtr = Xtr_df[cols].to_numpy()
        Xte = Xte_df[cols].to_numpy()
        bundle_cfg[skey] = {"modalities": mods, "features": cols, "targets": {}}
        sk_models[skey] = {}
        ablation[skey] = {}
        test_metrics[skey] = {}
        conformal_report[skey] = {}
        for tgt in TARGETS:
            fam = chosen[tgt]
            C = C_of.get(tgt)
            y = Ytr[tgt].to_numpy()
            yt = Yte[tgt].to_numpy()
            s_oof, aucs = oof_scores(fam, Xtr, y, C=C, cols=cols)
            a, b = fit_platt(s_oof, y)
            p_oof = sigmoid(a * s_oof + b)
            q = conformal_thresholds(p_oof, y)
            main_m = make_model(fam, C=C).fit(Xtr, y)
            boots = []
            for bi in range(N_BOOT_MODELS):
                bidx = np.random.default_rng(SEED + 100 + bi).integers(0, len(y), len(y))
                if len(np.unique(y[bidx])) < 2:
                    continue
                boots.append(make_model(fam, C=C).fit(Xtr[bidx], y[bidx]))
            p_te = sigmoid(a * raw_score(main_m, Xte) + b)
            s0, s1 = conformal_set(p_te, q)
            covered = np.where(yt == 1, s1, s0)
            sizes = s0.astype(int) + s1.astype(int)
            ablation[skey][tgt] = {"cv_auc_mean": round(float(aucs.mean()), 4),
                                   "cv_auc_std": round(float(aucs.std()), 4)}
            test_metrics[skey][tgt] = metrics_with_ci(yt, p_te, rng) if skey == "ecg+labs+echo" else \
                {k: round(float(v), 4) for k, v in point_metrics(yt, p_te).items()}
            conformal_report[skey][tgt] = {"alpha": ALPHA, "target_coverage": 1 - ALPHA,
                                           "test_coverage": round(float(covered.mean()), 4),
                                           "share_single": round(float((sizes == 1).mean()), 4),
                                           "share_both": round(float((sizes == 2).mean()), 4),
                                           "share_empty": round(float((sizes == 0).mean()), 4)}
            sk_models[skey][tgt] = {"main": main_m, "boot": boots, "platt": (a, b), "q": q, "cols": cols}
            bundle_cfg[skey]["targets"][tgt] = {
                "family": fam, "platt": [a, b], "conformal": {"q0": q[0], "q1": q[1], "alpha": ALPHA},
                "model": export_model(main_m, cols), "boot": [export_model(m, cols) for m in boots],
            }
            if skey == "ecg+labs+echo":
                oof_store[tgt] = (p_oof, y)
            log(skey, tgt, fam, "cvAUC", ablation[skey][tgt]["cv_auc_mean"],
                "testAUC", round(float(roc_auc_score(yt, p_te)), 3), "cov", conformal_report[skey][tgt]["test_coverage"])

    # ---------------------------------------------------------------- 3. reports on the full set
    FULL = "ecg+labs+echo"
    reports = {"calibration": {}, "curves": {}, "decision_curve": {}, "subgroups": {}}
    Xte_full = Xte_df[full_cols].to_numpy()
    for tgt in TARGETS:
        p_oof, y = oof_store[tgt]
        mdl = sk_models[FULL][tgt]
        a, b = mdl["platt"]
        p_te = sigmoid(a * raw_score(mdl["main"], Xte_full) + b)
        yt = Yte[tgt].to_numpy()
        reports["calibration"][tgt] = {"oof": calibration_bins(y, p_oof), "brier_oof": round(float(brier_score_loss(y, p_oof)), 4)}
        reports["curves"][tgt] = {"test": curve_points(yt, p_te), "oof": curve_points(y, p_oof)}
        reports["decision_curve"][tgt] = decision_curve(y, p_oof)
        sub = {}
        sex = Xtr_df["Sex"].to_numpy()
        age = Xtr_df["Age"].to_numpy()
        for name, m in [("male", sex == 1), ("female", sex == 0), ("age<60", age < 60), ("age>=60", age >= 60)]:
            if len(np.unique(y[m])) == 2:
                sub[name] = {"n": int(m.sum()), "auc": round(float(roc_auc_score(y[m], p_oof[m])), 4),
                             "prevalence": round(float(y[m].mean()), 4)}
        reports["subgroups"][tgt] = sub

    # ---------------------------------------------------------------- 4. OOD statistics (full set, continuous)
    cont = [f["name"] for f in F.feature_meta(X_all) if f["kind"] == "numeric"]
    Z = StandardScaler().fit(Xtr_df[cont])
    Ztr = Z.transform(Xtr_df[cont])
    lw = LedoitWolf().fit(Ztr)
    prec = lw.precision_
    d_tr = np.sqrt(np.einsum("ij,jk,ik->i", Ztr - lw.location_, prec, Ztr - lw.location_))
    ood = {"features": cont, "mean": Z.mean_.tolist(), "scale": Z.scale_.tolist(),
           "location": lw.location_.tolist(), "precision": prec.round(10).tolist(),
           "threshold": float(np.percentile(d_tr, 99)),
           "ranges": {c: [float(Xtr_df[c].min()), float(Xtr_df[c].max())] for c in full_cols}}

    # ---------------------------------------------------------------- 5. presets (real held-out patients)
    mdl = sk_models[FULL]["CAD"]
    a, b = mdl["platt"]
    p_cad_te = sigmoid(a * raw_score(mdl["main"], Xte_full) + b)
    nv_te = F.nvessels(Yte)
    # only typical (in-distribution) patients are used as demo presets
    Zte = Z.transform(Xte_df[cont]) - lw.location_
    d_te = np.sqrt(np.einsum("ij,jk,ik->i", Zte, prec, Zte))
    in_range = np.all([(Xte_df[c] >= ood["ranges"][c][0]) & (Xte_df[c] <= ood["ranges"][c][1]) for c in full_cols], axis=0)
    typical = (d_te <= ood["threshold"]) & in_range
    order = [i for i in np.argsort(p_cad_te) if typical[i]]
    low = next(i for i in order if Yte["CAD"].iloc[i] == 0)
    high = next((i for i in order[::-1] if nv_te[i] == 3), order[-1])
    mid = min(order, key=lambda i: abs(p_cad_te[i] - 0.55))
    presets = []
    for label, i in [("Low risk (held-out patient)", low), ("Borderline (held-out patient)", mid),
                     ("High risk, 3-vessel (held-out patient)", high)]:
        presets.append({"label": label, "values": {c: float(Xte_df[c].iloc[i]) for c in full_cols},
                        "truth": {t: int(Yte[t].iloc[i]) for t in TARGETS}})

    # ---------------------------------------------------------------- 6. parity fixture (python reference outputs)
    parity = {"X": Xte_df[full_cols].to_numpy().tolist(), "set": FULL, "expected": {}}
    for tgt in TARGETS:
        m = sk_models[FULL][tgt]
        a, b = m["platt"]
        parity["expected"][tgt] = sigmoid(a * raw_score(m["main"], Xte_full) + b).round(10).tolist()

    meta = F.feature_meta(X_all)
    bundle = {
        "version": time.strftime("%Y-%m-%d %H:%M"),
        "data_sha256_12": data_hash,
        "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "targets": TARGETS,
        "modalities": F.MODALITIES, "modality_label": F.MODALITY_LABEL,
        "optional_modalities": F.OPTIONAL_MODALITIES,
        "features": meta,
        "whatif_targets": F.WHATIF_TARGETS,
        "configs": bundle_cfg,
        "ood": ood,
        "presets": presets,
        "metrics": {"test_full": test_metrics[FULL], "benchmark": benchmark, "ablation": ablation,
                    "conformal": conformal_report, "test_by_set": test_metrics},
        "reports": reports,
    }
    (ART / "model_bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    js = "window.MODEL_BUNDLE = " + json.dumps(bundle, separators=(",", ":")) + ";\n"
    (ART / "model_bundle.js").write_text(js)
    (ROOT / "web" / "data").mkdir(exist_ok=True)
    (ROOT / "web" / "data" / "model_bundle.js").write_text(js)  # served to the browser engine
    (REP / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=1))
    (REP / "reports.json").write_text(json.dumps(reports, indent=1))
    (REP / "parity_fixture.json").write_text(json.dumps(parity))
    joblib.dump({"models": sk_models, "full_cols": full_cols, "background_mean": Xtr_df[full_cols].mean().to_dict(),
                 "train_X": Xtr_df[full_cols]}, ART / "models.joblib")
    log(f"done in {time.time() - t0:.0f}s; bundle {(ART / 'model_bundle.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
