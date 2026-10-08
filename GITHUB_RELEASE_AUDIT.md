# GitHub local release audit

## 1. Release tree
Local review only. No remote repository was created and no upload was performed.
File count: 34. Upload repository contents while retaining subdirectories; do not upload the separate ZIP as a substitute for extracted source files.

```text
.gitignore
CITATION.cff.template
GITHUB_RELEASE_AUDIT.md
LICENSE
README.md
code/build_episodes.py
code/config.py
code/evaluate.py
code/figures.py
code/identity_linkage.py
code/modeling.py
code/predict.py
code/preprocess.py
code/run_analysis.py
code/shap_analysis.py
code/supplementary_tables.py
code/tables.py
code/validate_inputs.py
example_data/synthetic_example.csv
metadata/candidate_features.json
metadata/cohort_definition.md
metadata/data_dictionary.csv
metadata/export_verification.json
metadata/feature_definitions.md
metadata/input_contract.md
metadata/model_card.md
metadata/source_code_provenance.json
metadata/workflow_map.md
model/feature_order.json
model/final_model_metadata.json
model/final_rf13_model.pkl
outputs_example/README.md
outputs_example/synthetic_predictions.csv
requirements.txt
```

## 2. Core scripts and reproducibility boundary
`metadata/workflow_map.md` maps all 13 modules to the actual frozen v2 implementation. `modeling.py` contains grouped CV, seven classifiers, adaptive tuning, ranking, one-SE selection, final fitting, all 40 single-variable comparators, strict-baseline sensitivity, freeze and external validation. `evaluate.py` contains patient-mean metrics, calibration and bootstrap inference. `tables.py`, `supplementary_tables.py`, and `figures.py` reproduce the source table/figure code, including post-hoc permutation SHAP. `predict.py` is the tested minimal inference entry point.

The public normalized-panel preprocessing adapter is source-based but requires prior custodian identity/eligibility adjudication. Private hospital extraction, administrative mappings and original files are deliberately not shipped. A fully unattended raw-hospital-export-to-results run is therefore not independently possible from this public package. The preferred exact-analysis route uses authorized frozen analysis tables with original anonymous grouping values and row order preserved. See the input contract.

## 3. Actual trained model
`model/final_rf13_model.pkl`: extracted only the final RandomForest pipeline from the frozen multi-model container, without fitting. The multi-model container is not released because other estimator types can retain training observations. The released object has a 13-variable median imputer, scaling passthrough and a 120-tree forest with minimum leaf size 2. Class order: [0, 1].

SHA-256: `0d73d17ca4375ac03ed7348d98f431d43b36fe84b88aaadc443dd90759300ab5`.

## 4. Ordered features and units
PASS: frozen container feature indices, candidate ordering, frozen JSON feature order, fitted input dimension and revision packet agree. The exact ordered packet sequence is consistent; no mismatch was silently changed. Case-source units agree with table definitions; independent control-unit provenance remains unverified.

| Order | Python column | Unit |
|---:|---|---|
| 1 | `LYMPH#` | 10⁹/L |
| 2 | `TP` | g/L |
| 3 | `GLB` | g/L |
| 4 | `HCT` | % |
| 5 | `LYMPH%` | % |
| 6 | `RDW-CV` | % |
| 7 | `NEUT%` | % |
| 8 | `RDW-SD` | fL |
| 9 | `HGB` | g/L |
| 10 | `MONO%` | % |
| 11 | `A/G` | ratio |
| 12 | `RBC` | 10¹²/L |
| 13 | `MCHC` | g/L |

All 40 candidates are defined in `metadata/data_dictionary.csv`; AGE/SEX are descriptive and not included among them.

## 5. Locked cutoff
**0.3873057203557378**. Derived by Youden index from full-development adaptive-procedure OOF probabilities averaged per person. It is not an externally optimized cutoff and not an RF13-only nested OOF estimate. Decision uses probability greater than or equal to cutoff.

## 6. Environment
Python **3.12.14**, scikit-learn **1.9.1**, seed **20261007**. Direct dependencies match the recorded original environment. A new isolated venv was created with Python 3.12.14 and the exact pinned requirements installed. The ambient system Python had different versions and was not used for final export or validation.

## 7. Real clinical data present?
**NO** real participant rows or administrative identifiers are included. The only row-data files are eight synthetic inference rows and their synthetic predictions. No original Excel, linkage mapping, salt, individual clinical prediction file, sample-level SHAP archive, old seven-feature model, Streamlit application or DCA workflow is present. Trained forest thresholds and imputation statistics are model parameters, not a released training matrix. This is not a formal differential-privacy guarantee.

## 8. Privacy and path scan
PASS: recursive text and model-string inspection found no prohibited identifiers or paths. Scan covered CJK text patterns, long identity-number/telephone patterns, administrative-identifier labels, drive-rooted paths, local usernames/source directories, pickle string opcodes and 9218 known private name/grouping/administrative tokens, without writing those tokens to the release. The standalone pickle has 93 string opcodes. Binary numeric arrays were treated as model parameters, not decoded as text. No unreviewed binary container is shipped. Final packaged-file hashes are checked against the scanned release.

## 9. Synthetic tests
PASS: 8 synthetic rows predicted in the clean environment; export predictions equal the original frozen pipeline on the same synthetic inputs. Reordered input columns are selected by the published feature order. Saved CSV probabilities differ only by floating-point text roundoff (maximum 5.55e-17). Running from an unrelated working directory succeeds.

Synthetic behavior checks also verified disjoint grouped folds, training-only imputation medians, nonconflicting identity bridging, separation of centers, retention of repeated panels, and exclusion of test-date proxies from strict baseline. Synthetic preprocessing produced 13 primary and 11 baseline rows per center in an internal scratch copy. Those test records are not included as clinical data or study results.

## 10. Installation, syntax, imports and paths
PASS: `pip install -r requirements.txt` and `pip check` completed in the new environment. All 13 scripts pass compilation and imports; imports create no output files. All dependency modules import. Paths are repository-relative and do not depend on the original machine. Full real-data training, bootstrap analysis and manuscript figure rendering were **not rerun**. Import/syntax checks do not establish full numerical equivalence of every adapted workflow stage. Frozen source code and original model hashes remain unchanged.

## 11. Methods and TRIPOD+AI support
This is a source/packet consistency assessment, not a certification of reporting-guideline compliance. The latest revised manuscript was not available for paragraph-by-paragraph comparison; original submission manuscripts are not treated as the current revision.

| Reporting item | Public support | Boundary |
|---|---|---|
| Participant flow | Cohort definitions and frozen figure code | Original linkage/eligibility evidence remains private |
| Predictor definition | All 40 names, units, roles and caveats | BUN terminology and control units unresolved |
| Missing data | Fold-local median imputer and serialized final statistics | No public original missingness rows |
| Model specification | Actual RF13 pipeline, full forest parameters, order and cutoff | Exact software versions required |
| Internal validation | 5 x 3 stratified grouped adaptive CV | OOF is adaptive, not identical RF13 in each fold |
| External validation | Freeze precedes external predictor-table access in modeling | Eligibility/identity checks necessarily precede analysis |
| Performance metrics | Patient mean predictions, AUROC/AP, 2,000 patient bootstraps | Intervals condition on fitted predictions |
| Calibration | Joint intercept/slope, CITL, Brier | Evaluation only; no external recalibration |
| Model availability | Tested trained object and synthetic command | Public-only data cannot regenerate study estimates |

## 12. Manuscript review items; no manuscript edited
No feature-order or cutoff discrepancy was found against frozen metadata and the revision packet. The following statements must be checked in the final Methods, legends and data-availability text:

1. Describe established-case versus healthy-control discrimination; do not claim screening or symptomatic differential diagnosis.
2. Distinguish adaptive nested OOF from the whole-development frozen RF13; outer-fold algorithms and feature counts vary.
3. State unweighted repeated-panel training, group-disjoint folds and patient-mean evaluation. Do not claim independent repeated rows for confidence intervals.
4. Define strict baseline as earliest eligible within included sources with reliable sampling chronology, not first-ever, nearest diagnosis or proven steroid-free. Sensitivity models are fitted separately.
5. Report imputation/scaling inside folds, grouped SVM calibration, RF ranking plus one-SE selection, and post-hoc permutation SHAP (40 background / 200 explained / five cycles). Source figure code's obsolete intermediate legend strings were overridden by its final replacement logic; the effective final values are 40 and 200.
6. Report the development-derived cutoff, 2,000 fixed-prediction patient bootstraps, AP definition of AUPRC, joint calibration and evaluation-only external calibration.
7. Preserve caveats about pretreatment attestation, date proxies, BUN terminology and control-unit provenance.
8. State that trained model, code, dictionary and synthetic example are public; controlled clinical records are not. Do not claim public data reproduce study metrics independently.
9. Confirm author/citation details and reuse license. The citation template is deliberately not active CFF metadata until verified authors are supplied. The official [CFF schema](https://raw.githubusercontent.com/citation-file-format/citation-file-format/main/schema.json) requires authors. No authors or DOI were invented. LICENSE currently grants no additional reuse rights; it is not an open-source license.

## 13. Suggested repository name
`dlbcl-healthy-control-reproducibility`

## 14. Suggested repository description
Frozen RF13 model and reproducible patient-grouped analysis for DLBCL-versus-healthy-control discrimination, with a 40-predictor dictionary and synthetic inference example; no clinical records included.

## Release status
Technical inference/import checks and privacy scan passed. Prepared for the owner's manual review, with citation and licensing explicitly pending. No remote publication, manuscript changes, retraining of study models or changes to locked results were performed.
