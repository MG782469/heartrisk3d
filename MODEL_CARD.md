# Model card: HeartRisk 3D

## Intended use
- **Purpose:** education and clinical decision *support*: estimate the probability of coronary artery disease (CAD) and of >=50% stenosis in the LAD, LCX and RCA from routine history, examination, ECG, laboratory and echocardiography data, and show where on the heart that risk is.
- **Users:** clinicians, students and researchers who understand the limitations below.
- **Not for:** diagnosis, ruling out CAD, treatment decisions, or use without clinical judgement. It does not replace coronary angiography.

## Data
- Extension of the Z-Alizadeh Sani dataset (UCI ML Repository, CC BY 4.0): 303 patients from one centre who underwent angiography; 55 input features after encoding; 4 targets (CAD 216/303, LAD 177, LCX 119, RCA 114).
- Split: stratified 80/20 (242 train, 61 held-out test), stratified on the number of diseased vessels, seed 42.
- See [data/DATA_CARD.md](data/DATA_CARD.md).

## Models
| Target | Deployed family | Why |
|---|---|---|
| CAD | Elastic-net logistic regression (C = 0.1) | Within 1 SE of the best of 4 families |
| LAD | Elastic-net logistic regression (C = 0.03) | Within 1 SE of the best |
| LCX | Gradient boosting (150 trees, depth 2) | Logistic regression was more than 1 SE below the best |
| RCA | Elastic-net logistic regression (C = 0.03) | Best family |

A classifier chain for LCX/RCA (adding CAD and LAD scores as inputs, nested in CV) gave no gain (LCX -0.004 ± 0.007, RCA +0.006 ± 0.009 AUC) and is not used.

Benchmarked families: elastic-net logistic regression (C tuned by nested CV), gradient boosting, random forest, monotone-constrained histogram gradient boosting. Probabilities are Platt-calibrated on out-of-fold scores. Eight versions of every model are trained, one for each combination of ECG / labs / echo availability.

## Performance
Held-out test (n = 61), 95% bootstrap CI:

| Target | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CAD | 0.853 [0.76-0.94] | 0.770 [0.66-0.87] | 0.875 [0.76-0.97] | 0.795 [0.67-0.91] | 0.833 [0.73-0.92] |
| LAD | 0.791 [0.67-0.89] | 0.754 [0.64-0.85] | 0.788 [0.63-0.92] | 0.765 [0.62-0.91] | 0.776 [0.64-0.88] |
| LCX | 0.756 [0.62-0.87] | 0.672 [0.54-0.79] | 0.684 [0.46-0.89] | 0.481 [0.30-0.68] | 0.565 [0.38-0.72] |
| RCA | 0.677 [0.53-0.82] | 0.672 [0.56-0.79] | 0.550 [0.33-0.77] | 0.500 [0.29-0.70] | 0.524 [0.32-0.69] |

Repeated 5x5 cross-validation on the training set: CAD 0.946 ± 0.038, LAD 0.853 ± 0.057, LCX 0.728 ± 0.066, RCA 0.743 ± 0.074 ROC-AUC. The gap between CV and test for CAD (0.95 vs 0.85) is within the uncertainty of a 61-patient test set but is a reminder that estimates from this dataset are noisy.

## Uncertainty and safety mechanisms
- **Conformal prediction** (Mondrian, alpha = 0.10): test coverage CAD 84%, LAD 82%, LCX 85%, RCA 82% (nominal 90%; sampling error about ±7 points). 13% of test patients get the "uncertain" set for CAD. The most probable label is always included in the set (sets only grow, so coverage holds).
- **Bootstrap band:** 10th-90th percentile over 10 refits.
- **Out-of-distribution flag:** any value outside the training range or Mahalanobis distance above the 99th training percentile.
- **Coherence check:** warns when vessel and CAD estimates disagree.

## Explanations
Exact linear SHAP (log-odds, relative to the average training patient) for the logistic models; TreeSHAP for gradient boosting (API) or occlusion-to-mean (browser). Attributions are associations, not causes.

## Known limitations and risks
- Single centre, 303 patients, no external validation; may not transfer to other populations or hospitals.
- LCX and RCA predictions are weak (AUC about 0.7); treat them as low-confidence.
- Confounding: some healthy changes (e.g. lower BMI) raise the model's estimate. The what-if view is labelled as model sensitivity, not treatment effect.
- Subgroup CV AUC is lower for age >= 60 (0.84) than for age < 60 (0.96) for CAD; groups are small.
- `Region RWMA` is a 0-4 score, not a location; it is never drawn onto the heart. The 3D heart and territories are illustrative.

## Ethics
No personal data leaves the device; the app stores nothing. A visible disclaimer is shown on every screen.
