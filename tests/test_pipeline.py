"""Automated checks: leakage, anatomy mapping, API contract, Python/JS parity and clinical sanity.

Run:  python -m pytest -q
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import features as F  # noqa: E402

BUNDLE = json.loads((ROOT / "artifacts" / "model_bundle.json").read_text())


def test_targets_never_used_as_inputs():
    raw = pd.read_csv(ROOT / "data" / "zalizadeh_sani_ext.csv")
    X = F.encode(raw)
    assert not set(F.LEAKAGE_COLUMNS) & set(X.columns)
    for cfg in BUNDLE["configs"].values():
        assert not set(F.LEAKAGE_COLUMNS) & set(cfg["features"])


def test_modality_sets_drop_the_right_features():
    for key, cfg in BUNDLE["configs"].items():
        mods = set(cfg["modalities"])
        assert {F.MODALITY_OF[f] for f in cfg["features"]} <= mods, key


def test_each_predicted_vessel_maps_to_exactly_one_3d_structure():
    js = (ROOT / "web" / "js" / "anatomy.js").read_text()
    targets = re.findall(r'target:\s*"(\w+)"', js)
    vessel_targets = [t for t in BUNDLE["targets"] if t != "CAD"]
    assert sorted(targets) == sorted(vessel_targets)


@pytest.fixture(scope="module")
def client():
    from app.server import app
    return app.test_client()


def test_api_predict_contract(client):
    preset = BUNDLE["presets"][1]["values"]
    r = client.post("/api/predict", json={"values": preset, "available": {"ecg": True, "labs": True, "echo": True}})
    assert r.status_code == 200
    out = r.get_json()
    for t in BUNDLE["targets"]:
        tt = out["targets"][t]
        assert 0 <= tt["p"] <= 1
        assert tt["band"][0] <= tt["band"][1]
        assert tt["drivers"]
    assert out["decision"]["level"] in {"high", "low", "unsure", "check"}


def test_api_matches_exported_bundle(client):
    fx = json.loads((ROOT / "artifacts" / "reports" / "parity_fixture.json").read_text())
    names = BUNDLE["configs"][fx["set"]]["features"]
    for i, row in enumerate(fx["X"][:15]):
        out = client.post("/api/predict", json={"values": dict(zip(names, row)),
                                                "available": {"ecg": True, "labs": True, "echo": True}}).get_json()
        for t in BUNDLE["targets"]:
            assert abs(out["targets"][t]["p"] - fx["expected"][t][i]) < 1e-9


def test_missing_modality_uses_reduced_model(client):
    preset = BUNDLE["presets"][1]["values"]
    out = client.post("/api/predict", json={"values": preset, "available": {"ecg": True, "labs": False, "echo": False}}).get_json()
    assert out["set"] == "ecg"
    assert all(d["modality"] in {"history", "symptoms", "ecg"} for d in out["targets"]["CAD"]["drivers"])


def test_clinical_sanity_typical_chest_pain_raises_cad_risk(client):
    v = dict(BUNDLE["presets"][1]["values"])
    full = {"ecg": True, "labs": True, "echo": True}
    v["Typical Chest Pain"] = 0
    lo = client.post("/api/predict", json={"values": v, "available": full}).get_json()["targets"]["CAD"]["p"]
    v["Typical Chest Pain"] = 1
    hi = client.post("/api/predict", json={"values": v, "available": full}).get_json()["targets"]["CAD"]["p"]
    assert hi > lo


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_engine_parity():
    r = subprocess.run(["node", str(ROOT / "tests" / "parity.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_conformal_set_always_contains_the_most_probable_label(client):
    full = {"ecg": True, "labs": True, "echo": True}
    fx = json.loads((ROOT / "artifacts" / "reports" / "parity_fixture.json").read_text())
    names = BUNDLE["configs"][fx["set"]]["features"]
    for row in fx["X"]:
        out = client.post("/api/predict", json={"values": dict(zip(names, row)), "available": full}).get_json()
        for t, r in out["targets"].items():
            top = ("CAD" if t == "CAD" else "Stenotic") if r["p"] >= 0.5 else ("No CAD" if t == "CAD" else "Normal")
            assert top in r["set"], (t, r["p"], r["set"])
