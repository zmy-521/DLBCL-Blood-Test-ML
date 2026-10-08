# Core workflow map

| Module | Purpose and frozen-source relationship |
|---|---|
| `config.py` | Relative paths and ordered candidate list; private runtime outputs in run/ |
| `identity_linkage.py` | Generic-key adaptation of reconstruct.py nonconflicting component bridges |
| `preprocess.py` | Local normalized-panel/control interface; center separation and anonymous grouping utility |
| `build_episodes.py` | Frozen build_episodes.py pairing and strict-baseline eligibility logic |
| `validate_inputs.py` | Added center, group, numeric and baseline-subset checks |
| `modeling.py` | Frozen seven-classifier grouped nested CV, inner tuning, RF ranking, one-SE selection, all 40 single-variable LR comparators, baseline sensitivity, development freeze then external prediction |
| `evaluate.py` | Frozen patient-mean metrics, AUROC/AP, calibration, Brier, 2,000 patient bootstraps and paired comparisons |
| `tables.py` | Frozen baseline and performance Word tables |
| `supplementary_tables.py` | Three table-generation blocks extracted from frozen packet.py |
| `figures.py` | Frozen figure workflow wrapped in main; includes exact permutation SHAP sampling and additivity checks |
| `shap_analysis.py` | Explicit entry point for that SHAP/figure workflow; also generates the figures |
| `run_analysis.py` | Fresh-run orchestration, blocks stale checkpoints |
| `predict.py` | Minimal inference using the actual released RF13 |

Separate numbered wrappers for each mathematical step are intentionally unnecessary: model selection and freeze/external ordering remain in one source-based workflow, so stages cannot accidentally use a different selection implementation. Function names `splits`, `estimator`, `tune`, `rank`, `feature_select`, `aggregate`, `threshold`, `calibration`, and `metrics` expose the individual operations for inspection.

Code portability edits: removed private absolute paths and bundled-dependency injection, renamed the anonymous grouping field, moved feature definitions to public metadata, normalized sex encoding, added safe entry points and input validation. No real-data training was performed during packaging. Full real-data numeric/plot equivalence after portability adaptation has not been executed; synthetic model inference equivalence was tested separately.
