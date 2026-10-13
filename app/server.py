"""HeartRisk 3D - Flask backend.

Serves the web app and a small JSON API backed by the trained scikit-learn models and SHAP.
Binds to 127.0.0.1 only: patient data never leaves the machine. Nothing is stored.

Run:  python app/server.py   ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUNDLE = json.loads((ROOT / "artifacts" / "model_bundle.json").read_text())
STORE = joblib.load(ROOT / "artifacts" / "models.joblib")
MODELS = STORE["models"]
TRAIN_X = STORE["train_X"]
META = {f["name"]: f for f in BUNDLE["features"]}
TARGETS = BUNDLE["targets"]
LABELS = {"CAD": ["No CAD", "CAD"], "LAD": ["Normal", "Stenotic"], "LCX": ["Normal", "Stenotic"], "RCA": ["Normal", "Stenotic"]}

app = Flask(__name__, static_folder=None)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def raw_score(model, X):
    return model.decision_function(X)


def set_key(available):
    on = [m for m in BUNDLE["optional_modalities"] if available and available.get(m)]
    return "+".join(on) if on else "bedside"


def vector(cols, values):
    x, imputed = [], []
    for c in cols:
        v = values.get(c)
        try:
            v = float(v)
            if math.isnan(v):
                raise ValueError
        except (TypeError, ValueError):
            v = META[c]["median"]
            imputed.append(c)
        x.append(v)
    return np.array([x]), imputed


@lru_cache(maxsize=64)
def explainer(skey, tgt):
    import shap  # imported lazily: start-up stays fast
    m = MODELS[skey][tgt]["main"]
    clf, scaler = m.named_steps["clf"], m.named_steps["scale"]
    bg = scaler.transform(TRAIN_X[MODELS[skey][tgt]["cols"]].to_numpy())
    if hasattr(clf, "coef_"):
        # mean background -> exact interventional SHAP for a linear model, in log-odds
        return "exact linear SHAP (shap.LinearExplainer, log-odds)", shap.LinearExplainer(clf, shap.maskers.Independent(bg, max_samples=bg.shape[0]))
    return "TreeSHAP (shap.TreeExplainer, path-dependent, log-odds)", shap.TreeExplainer(clf)


def attribution(skey, tgt, x):
    method, ex = explainer(skey, tgt)
    z = MODELS[skey][tgt]["main"].named_steps["scale"].transform(x)
    # float32 tree splits make the additivity check fail by ~1e-2 on boundary cases; values are still TreeSHAP
    sv = ex.shap_values(z, check_additivity=False) if method.startswith("TreeSHAP") else ex.shap_values(z)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return method, sv.reshape(-1)


def ood(values):
    o = BUNDLE["ood"]
    out_of_range = []
    for name, (lo, hi) in o["ranges"].items():
        try:
            v = float(values.get(name))
        except (TypeError, ValueError):
            continue
        if v < lo or v > hi:
            out_of_range.append({"name": name, "value": v, "range": [lo, hi]})
    z = np.array([(float(values.get(n, META[n]["median"])) - o["mean"][i]) / o["scale"][i] - o["location"][i]
                  for i, n in enumerate(o["features"])])
    d = float(math.sqrt(max(z @ np.array(o["precision"]) @ z, 0.0)))
    return {"outOfRange": out_of_range, "distance": d, "threshold": o["threshold"],
            "isOOD": d > o["threshold"] or bool(out_of_range)}


def decision(out):
    c = out["targets"]["CAD"]
    if out["ood"]["isOOD"]:
        return {"level": "check", "title": "Check the inputs", "text": "Some values are outside what the model was trained on, so the estimate is less reliable."}
    k = c["setKind"]
    if k == "positive":
        return {"level": "high", "title": "CAD likely", "text": "At 90% confidence the model's prediction set contains only CAD. Consider cardiology review / angiography as per clinical judgement."}
    if k == "negative":
        return {"level": "low", "title": "CAD unlikely", "text": "At 90% confidence the prediction set contains only 'No CAD'. This does not rule out disease; use clinical judgement."}
    if k == "both":
        return {"level": "unsure", "title": "Uncertain: refer", "text": "The model cannot separate CAD from no CAD with 90% confidence for this patient. Refer to a cardiologist."}
    return {"level": "check", "title": "Atypical pattern", "text": "No label reaches 90% confidence (empty set). Re-check the inputs."}


def predict(values, available):
    skey = set_key(available)
    cfg = BUNDLE["configs"][skey]
    cols = cfg["features"]
    x, imputed = vector(cols, values)
    out = {"set": skey, "modalities": cfg["modalities"], "imputed": imputed, "targets": {}}
    for tgt in TARGETS:
        m = MODELS[skey][tgt]
        a, b = m["platt"]
        p = float(sigmoid(a * raw_score(m["main"], x) + b)[0])
        allp = [float(sigmoid(a * raw_score(bm, x) + b)[0]) for bm in m["boot"]] + [p]
        q = m["q"]
        # Mondrian set plus the most probable label (only enlarges the set: coverage still holds)
        in1, in0 = ((1 - p) <= q[1]) or p >= 0.5, (p <= q[0]) or p < 0.5
        method, contrib = attribution(skey, tgt, x)
        drivers = sorted(({"name": c, "label": META[c]["label"], "modality": META[c]["modality"],
                           "value": float(x[0, i]), "contrib": float(contrib[i])} for i, c in enumerate(cols)),
                         key=lambda d: -abs(d["contrib"]))
        by_mod, total = {}, sum(abs(d["contrib"]) for d in drivers)
        for d in drivers:
            e = by_mod.setdefault(d["modality"], {"signed": 0.0, "abs": 0.0})
            e["signed"] += d["contrib"]
            e["abs"] += abs(d["contrib"])
        for e in by_mod.values():
            e["share"] = e["abs"] / total if total else 0.0
        out["targets"][tgt] = {
            "p": p, "band": [float(np.percentile(allp, 10)), float(np.percentile(allp, 90))],
            "set": [l for l, keep in zip(LABELS[tgt], (in0, in1)) if keep],
            "setKind": "both" if in0 and in1 else "positive" if in1 else "negative" if in0 else "empty",
            "family": BUNDLE["configs"][skey]["targets"][tgt]["family"],
            "drivers": drivers, "modalityContribution": by_mod, "attribution_method": method,
        }
    T = out["targets"]
    out["expectedVessels"] = T["LAD"]["p"] + T["LCX"]["p"] + T["RCA"]["p"]
    mx = max(T["LAD"]["p"], T["LCX"]["p"], T["RCA"]["p"])
    out["coherence"] = {"ok": not (mx > T["CAD"]["p"] + 0.25 or (T["CAD"]["p"] > 0.8 and mx < 0.3)),
                        "note": "CAD in this dataset means at least one of LAD/LCX/RCA has >=50% stenosis."}
    out["ood"] = ood(values)
    out["decision"] = decision(out)
    return out


def what_if(values, available):
    base = predict(values, available)
    rows = []
    h = float(values.get("Length") or 0) / 100
    for name, target in BUNDLE["whatif_targets"].items():
        if target is None:
            continue
        cur = float(values.get(name))
        if not (cur < target if name == "HDL" else cur > target):
            continue
        v2 = dict(values, **{name: target})
        if name == "BMI" and h > 0:
            v2["Weight"] = round(target * h * h, 1)
        r = predict(v2, available)
        rows.append({"name": name, "label": META[name]["label"], "from": cur, "to": target,
                     "delta": {t: r["targets"][t]["p"] - base["targets"][t]["p"] for t in TARGETS}})
    rows.sort(key=lambda r: r["delta"]["CAD"])
    return {"base": base, "rows": rows}


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return jsonify(status="ok", version=BUNDLE["version"], data=BUNDLE["data_sha256_12"])


@app.get("/api/schema")
def schema():
    return jsonify(features=BUNDLE["features"], modalities=BUNDLE["modalities"], optional=BUNDLE["optional_modalities"])


@app.get("/api/metrics")
def metrics():
    return jsonify(metrics=BUNDLE["metrics"], reports=BUNDLE["reports"])


@app.get("/api/patients")
def patients():
    return jsonify(BUNDLE["presets"])


@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True) or {}
    return jsonify(predict(body.get("values", {}), body.get("available", {})))


@app.post("/api/whatif")
def api_whatif():
    body = request.get_json(force=True) or {}
    return jsonify(what_if(body.get("values", {}), body.get("available", {})))


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(ROOT / "web", path)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("Preparing SHAP explainers ...", flush=True)
    for skey in BUNDLE["configs"]:
        for tgt in TARGETS:
            explainer(skey, tgt)
    print(f"HeartRisk 3D running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False)
