"""Re-select demo presets in an existing bundle (same rule as train.py) without retraining."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import features as F
from sklearn.model_selection import train_test_split
b = json.loads((ROOT/"artifacts/model_bundle.json").read_text())
raw = pd.read_csv(ROOT/"data/zalizadeh_sani_ext.csv"); X = F.encode(raw); Y = F.labels(raw)
_, te = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=F.RANDOM_SEED, stratify=F.nvessels(Y))
Xte, Yte = X.iloc[te], Y.iloc[te]
o = b["ood"]
Z = (Xte[o["features"]].to_numpy() - np.array(o["mean"])) / np.array(o["scale"]) - np.array(o["location"])
d = np.sqrt(np.einsum("ij,jk,ik->i", Z, np.array(o["precision"]), Z))
inr = np.all([(Xte[c] >= lo) & (Xte[c] <= hi) for c, (lo, hi) in o["ranges"].items()], axis=0)
typ = (d <= o["threshold"]) & inr
cfg = b["configs"]["ecg+labs+echo"]["targets"]["CAD"]; m = cfg["model"]; a, bb = cfg["platt"]
z = (Xte[F.FEATURE_NAMES].to_numpy() - np.array(m["mean"])) / np.array(m["scale"])
p = 1 / (1 + np.exp(-(a * (z @ np.array(m["coef"]) + m["intercept"]) + bb)))
nv = F.nvessels(Yte)
order = [i for i in np.argsort(p) if typ[i]]
low = next(i for i in order if Yte["CAD"].iloc[i] == 0)
high = next((i for i in order[::-1] if nv[i] == 3), order[-1])
mid = min(order, key=lambda i: abs(p[i] - 0.55))
print("typical", typ.sum(), "of", len(typ), "picked", low, mid, high, p[[low, mid, high]].round(3), d[[low, mid, high]].round(2), o["threshold"])
b["presets"] = [{"label": l, "values": {c: float(Xte[c].iloc[i]) for c in F.FEATURE_NAMES}, "truth": {t: int(Yte[t].iloc[i]) for t in b["targets"]}}
                for l, i in [("Low risk (held-out patient)", low), ("Borderline (held-out patient)", mid), ("High risk, 3-vessel (held-out patient)", high)]]
js = json.dumps(b, separators=(",", ":"))
(ROOT/"artifacts/model_bundle.json").write_text(js)
(ROOT/"artifacts/model_bundle.js").write_text("window.MODEL_BUNDLE = " + js + ";\n")
(ROOT/"web/data/model_bundle.js").write_text("window.MODEL_BUNDLE = " + js + ";\n")
