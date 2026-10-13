# Demo video script (target 5-6 min, allowed 3-10 min)

Setup before recording
- `python app/server.py` and open http://127.0.0.1:8000 in full-screen Chrome/Edge (F11), zoom 100%.
- Record with OBS or Windows Game Bar (Win + G), 1080p, microphone on. Close notifications.
- Useful deep links: `?preset=0`, `?preset=1`, `?preset=2`, `?preset=1&off=labs,echo`, `?preset=1&tab=model`.

| Time | Screen | What to say (English, speak slowly) |
|---|---|---|
| 0:00-0:25 | App opens on the high-risk patient, heart beating | "Hi, I'm Manas from BIT Mesra. This is HeartRisk 3D for Track A. A risk percentage alone doesn't tell a doctor *where* the disease is. HeartRisk 3D predicts coronary artery disease and stenosis of the LAD, LCX and RCA, and shows it on an interactive 3D heart." |
| 0:25-0:50 | Point at the disclaimer and privacy badges | "Two things first: this is decision support and education only, not a diagnosis. And no data leaves the device: it runs on localhost, or fully in the browser on the live demo." |
| 0:50-1:40 | Rotate, zoom, click LAD, then camera buttons Anterior / Left lateral / Inferior | "Each artery is coloured by its predicted probability: teal low, amber medium, crimson high. The arteries follow real anatomy: LAD down the front groove to the apex, LCX around the left side, RCA on the right down to the posterior descending artery. The left main is grey because it is not predicted. Clicking a vessel opens its card. Note: the dataset says whether a vessel is narrowed, not where, so we colour the whole vessel instead of inventing a lesion. Flow slows down as risk rises, and the bullseye shows the typical heart-wall territories." |
| 1:40-2:30 | Summary tab | "The summary gives a decision, not just a number: here 'CAD likely'. The probability has a band from 10 bootstrap models, and a 90% conformal prediction set. Expected diseased vessels is the sum of the three vessel probabilities: 2.7 of 3, which matches this patient's real triple-vessel disease." |
| 2:30-3:15 | Switch to Borderline preset, then untick Labs and Echo | "Now a borderline patient. Many primary health centres in India have no echo or lab results. When I switch them off, the app uses a model trained without them: confidence drops, the band widens, and the decision becomes 'Uncertain: refer to a cardiologist'. This is multimodal AI being honest about missing modalities." |
| 3:15-4:00 | Why tab, CAD then LAD | "Why did the model say this? Exact SHAP values for this patient: typical chest pain raises the risk, younger age lowers it. The top bar shows which test type drove the prediction: history, symptoms, ECG, labs or echo. Abnormal values are flagged against normal ranges." |
| 4:00-4:40 | What-if tab, move BP and BMI sliders | "What-if lets a clinician see how the model responds to modifiable factors. Look: lowering BMI actually raises the model's estimate. That is confounding in observational data, the 'obesity paradox', and the app says so. That's why we call this model sensitivity, never treatment advice." |
| 4:40-5:30 | Model tab, scroll | "Validation: leakage-free, targets are never inputs and a test enforces it. We compared four model families with repeated cross-validation and kept the simplest one within one standard error. On 61 held-out patients CAD ROC-AUC is 0.85 with confidence interval, LAD 0.79, while LCX and RCA are harder, around 0.7, and we show that openly. Calibration, decision curve, conformal coverage and subgroups are all here." |
| 5:30-6:00 | VS Code: repo tree, then run `python -m pytest -q` | "Technically: Python and scikit-learn for training, a Flask API with SHAP, and the same model bundle running in the browser with a parity test. Eight automated checks pass. Code, model weights and documentation are on GitHub. Thank you!" |

Tips
- If the Hindi audio voice is installed, click "हिंदी में सुनें" for 5 seconds during the Summary part; skip it if no Hindi voice is available.
- Keep the mouse slow; pause one second after every click.
- Upload to YouTube as **Public** or **Unlisted** and check the link in a private window before submitting.
