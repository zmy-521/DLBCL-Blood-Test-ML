# Reproducible DLBCL-versus-healthy-control discrimination

## Study title and purpose
Repository working title: **DLBCL-versus-healthy-control discrimination using routine laboratory measurements**.
This is the frozen revision-v2 analysis companion; the final bibliographic manuscript title and authors are not yet confirmed for this release.
The model distinguishes established DLBCL cases from healthy controls.
It is NOT validated for general-population screening.
It is NOT validated as a differential diagnostic tool in symptomatic patients.

## Repository contents
- `code/`: preprocessing, patient-grouped nested validation, seven classifiers, feature selection, evaluation, tables and figures.
- `model/`: the actual trained RF13 pipeline, ordered features and model metadata.
- `metadata/`: all 40 candidate definitions, cohort and input contracts, model card and source-code provenance.
- `example_data/`: eight fully synthetic rows.
- `outputs_example/`: synthetic inference output only.
- `GITHUB_RELEASE_AUDIT.md`: release checks and unresolved limitations.

## Cohort structure
| Cohort | DLBCL people / panels | Healthy controls / panels |
|---|---:|---:|
| Development primary | 65 / 285 | 430 / 430 |
| External primary | 131 / 240 | 248 / 248 |
| Development strict baseline | 65 / 65 | 430 / 430 |
| External strict baseline | 41 / 41 | 248 / 248 |

All included case specimens were investigator-confirmed pretreatment. Structured treatment, diagnosis and steroid timestamps were unavailable. Cases may contribute repeated panels; controls are investigator-attested distinct people. Chemistry and CBC pairing requires one of each within the same verified person and calendar day. See `metadata/cohort_definition.md`.

## Modeling overview and patient-grouped validation
Fixed seed: 20261007. Outer five-fold and inner three-fold stratified group CV keep all panels for a person together. Median imputation is fitted only on training folds. Standardization is used for LR, SVM and KNN only. SVM probability calibration uses group-disjoint splits.

Seven algorithms: logistic regression, random forest, XGBoost, SVM, KNN, Gaussian naive Bayes and decision tree. The adaptive procedure selects algorithm and parameters using inner mean patient AUROC, then selects the smallest feature count within one standard error of the best inner score. RF importance ranking is recomputed within training folds. SHAP is post hoc, not the selection method.

Training uses all eligible panels without inverse-frequency weighting. Evaluation averages panel probabilities per person. OOF results evaluate the adaptive selection procedure; different outer folds can select different algorithms and features. They do not evaluate one identical RF13 in every fold. Strict-baseline sensitivity reuses each outer training set's selection and fits on baseline-only records.

## External validation
External validation data were not used for tuning/model selection. The complete development workflow freezes all models and development-derived cutoffs before opening external predictor tables for one prediction pass. AUROC, average precision (AUPRC), classification metrics, Brier score, joint calibration intercept/slope and calibration-in-the-large are reported. Bootstrap intervals use 2,000 stratified patient resamples of fixed predictions; they do not include model-refitting uncertainty. External calibration is evaluation only.

## Final model
The released `model/final_rf13_model.pkl` is extracted without retraining from the frozen v2 model: 120 trees, minimum leaf size 2, median imputation, no scaling. Exact ordered columns and units are in `model/feature_order.json` and `model/final_model_metadata.json`.

Development-derived cutoff: **0.3873057203557378**, chosen from patient-mean adaptive-procedure OOF probabilities using Youden's index. The released forest was then fitted on all development panels. Do not change the cutoff based on external outcomes.

## Environment and quick synthetic example
Python 3.12.14 and scikit-learn 1.9.1; all direct dependencies are pinned in `requirements.txt`. Create a Python 3.12 environment and run from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python code/predict.py
```

The final command reads `example_data/synthetic_example.csv`, loads the trusted frozen model, and writes `outputs_example/synthetic_predictions.csv`. Only load pickle files from trusted sources.

**Synthetic demonstration data; not derived from individual study participants.** These eight rows demonstrate the inference interface only. They have no observed outcome labels and cannot reproduce performance estimates or support nested CV.

## How to reproduce the research analysis
Controlled-access data are required. The public repository alone reproduces synthetic inference, not the study estimates.

1. In a fresh copy, place the four approved anonymous analysis tables under `run/data_v2/` according to `metadata/input_contract.md`. Retain exact original row order and grouping tokens to reproduce original fold assignments. An optional normalized-panel route uses `python code/preprocess.py`; source-system extraction and eligibility adjudication remain with the data custodian.
2. Run `python code/validate_inputs.py`.
3. Run `python code/run_analysis.py --include-figures`.

The script fits development models, freezes them, predicts external cases, evaluates, and produces the original table/figure workflow including permutation SHAP. This may take substantial time. The release validation did **not** rerun this analysis on real records. Research outputs and checkpoints may contain private row-level information and are excluded by `.gitignore`; inspect any future upload manually. Do not upload `run/` or `local_inputs/`.

The figures contain the frozen study's cohort counts and assume the actual study inputs. They are not a generic chart template for arbitrary new cohorts. Original Times New Roman figure typography requires that font locally; otherwise Matplotlib may substitute a font. See `metadata/workflow_map.md` for individual modules.

## Data availability
**Patient-level clinical data are NOT included.** Original hospital exports, administrative linkage, exact collection dates, salts, row-level predictions and individual SHAP arrays are not distributed. Access to real study data requires applicable institutional authorization and a request to the study investigators; no unverified contact or access guarantee is supplied here.

## Limitations
Case-versus-healthy spectrum selection limits transportability and clinical interpretation. Repeated panels have unequal training influence. Some episode dates use test-date proxies. Strict baseline is earliest eligible within included sources, not necessarily nearest diagnosis, first-ever, or steroid-free. BUN source terminology and independent control-unit provenance remain unresolved. Missing values in four nonfinal platelet indices occur externally. No external recalibration or clinical-impact validation is claimed. See the model card.

## Citation and license
`CITATION.cff.template` is an explicitly unfinished citation template. Its working title is not a final bibliographic citation. Replace it with the verified manuscript citation and authors before linking a publication or DOI. No published citation is invented. Licensing status is stated in `LICENSE`; public visibility does not itself grant unrestricted reuse rights.
