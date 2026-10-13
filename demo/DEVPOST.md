# Devpost submission text (copy-paste)

**Project name:** HeartRisk 3D

**Tagline (short):** Coronary risk, mapped to the heart: per-vessel CAD prediction on an interactive 3D heart, with explanations, uncertainty and "uncertain: refer" safety.

**Track:** Track A: Cardiovascular Risk Visualization & Prediction

## Inspiration
A risk percentage alone does not tell a clinician or a patient where disease is likely or how far to trust the number. We wanted a tool that shows coronary risk on the heart itself, explains every prediction, and says honestly when it is unsure, especially for Indian primary health centres where echo or lab tests are often unavailable.

## What it does
- Predicts overall coronary artery disease (CAD) and >=50% stenosis of the LAD, LCX and RCA from 55 routine features (history, symptoms, ECG, labs, echo).
- Colours each artery of an interactive 3D heart by its probability (rotate, zoom, click, camera presets, flow that slows with risk, AHA 17-segment bullseye of typical territories).
- Explains each prediction with SHAP and shows which test type (history, symptoms, ECG, labs, echo) drove it.
- Gives an uncertainty band and a 90% conformal prediction set, which turns into a clear decision: CAD likely / CAD unlikely / Uncertain: refer.
- "Tests available" switch: if echo or labs were not done, a model trained without them is used and the loss of confidence is visible.
- What-if for modifiable factors, explicitly labelled as model sensitivity (it even flags the counter-intuitive "obesity paradox" in this data).
- One-click printable report and a guided tour for first-time users.
- Out-of-distribution warnings, coherence checks and a disclaimer on every screen. Data never leaves the device.

## How we built it
Python + scikit-learn: four model families compared with repeated 5x5 cross-validation, simplest within one standard error deployed (logistic regression for CAD/LAD/RCA, gradient boosting for LCX), Platt calibration on out-of-fold scores, Mondrian conformal prediction, 10 bootstrap refits, and 8 models per target for every combination of ECG/labs/echo. Flask API with SHAP; the same exported model bundle also runs in the browser (parity-tested to 5e-11) for a no-install live demo. Three.js procedural heart with anatomically placed coronaries driven by a single anatomy registry. 8 automated tests (leakage, anatomy mapping, API contract, parity, clinical sanity).

## Results (held-out test, n = 61, 95% CI)
CAD ROC-AUC 0.85 [0.76-0.94] (CV 0.95), LAD 0.79 [0.67-0.89], LCX 0.76 [0.62-0.87], RCA 0.68 [0.53-0.82]. Full metrics, calibration, decision curve, conformal coverage and subgroups are in the app.

## Challenges we ran into
Small data (303 patients) makes test estimates noisy, so we report CIs and cross-validation together. Two traps in the problem statement: Region RWMA is a score, not a location, and the data has no lesion position, so we never draw fake anatomical findings. Exact parity between Python and JavaScript tree models required full-precision thresholds.

## Accomplishments
Honest, explainable, uncertainty-aware predictions; a multimodal ablation that shows the value of each test type; a live demo that keeps patient data in the browser.

## What we learned
Calibration and "knowing when you don't know" matter more than a single accuracy number in medical AI, and observational models must not be read causally.

## What's next
External validation on another hospital's data, patient-specific anatomy from CT angiography, and clinician feedback.

**Built with:** python, scikit-learn, shap, flask, three.js, javascript, html, css, numpy, pandas

**Links:** GitHub repo · Live demo (GitHub Pages) · YouTube video · docs/DOCUMENTATION.pdf

**Disclaimer:** Decision support / education only; not a substitute for formal diagnostic imaging.
