#!/usr/bin/env sh
# HeartRisk 3D - start on macOS / Linux
cd "$(dirname "$0")"
[ -f artifacts/models.joblib ] || python3 ml/train.py
python3 app/server.py 8000
