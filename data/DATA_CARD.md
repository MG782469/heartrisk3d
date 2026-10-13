# Data card

- **Name:** Extension of Z-Alizadeh Sani dataset
- **Source:** UCI Machine Learning Repository, dataset 411 (https://archive.ics.uci.edu/dataset/411), downloaded 13 Oct 2026
- **Licence:** CC BY 4.0
- **File:** `zalizadeh_sani_ext.csv` (converted from the original `.xlsx`, unchanged values), 303 rows x 59 columns
- **Population:** patients evaluated for CAD at one centre, all of whom had coronary angiography
- **Targets:** `Cath` (CAD / Normal), `LAD`, `LCX`, `RCA` (Stenotic / Normal; stenosis >= 50%). CAD equals "at least one vessel stenotic" for 302 of 303 patients.
- **Inputs:** demographics and history (17), symptoms and examination (13), ECG (7, BBB one-hot to 2), laboratory (14), echocardiography (3). `Exertional CP` is constant and dropped.
- **Missing values:** none
- **Caveats:** `Region RWMA` is a 0-4 score without location. Measurements are single time-point. No external cohort is available in this project.
