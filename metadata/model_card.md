# Frozen RF13 model card

**Purpose:** DLBCL-versus-healthy-control discrimination.

**Warning:** Not validated for population screening or differential diagnosis among symptomatic patients.

The model is a random forest with 120 trees, minimum leaf size 2, seed 20261007, trained on development-only panels. Pipeline, full forest parameters, feature order, units, class encoding and cutoff are in `model/final_model_metadata.json`. A probability is a model output in the sampled case-control setting, not a calibrated estimate of population prevalence or a clinical diagnosis.

The development adaptive-procedure OOF AUROC was approximately 0.9628 and the external frozen RF13 AUROC approximately 0.9648. External logistic regression achieved approximately 0.9941, but was not selected after examining external performance. These rounded aggregate values describe the locked analysis, not metrics computed from the synthetic example. OOF and frozen external RF13 evaluate different estimation stages and should be described accordingly.

Median imputation is fitted within training folds; only LR/SVM/KNN use fold-local standardization. SVM sigmoid calibration uses grouped splits. Algorithm/parameter selection, RF importance ranking and one-SE feature count all exclude external data. Post-hoc permutation SHAP uses 40 development baseline people as background, 200 explained people, five cycles and a fixed seed; additivity is checked. No original individual explanations are released.

Patient bootstraps hold fitted predictions fixed. Joint calibration intercept/slope and slope-fixed-at-one calibration-in-the-large are evaluation summaries, not recalibration. Strict-baseline models are separately fitted sensitivity models and must not be called the same fitted RF13.

Limitations include case-control spectrum, unweighted repeated measurements, date proxies, retrospective eligibility attestations, incomplete unit provenance and uncertain BUN terminology. The export includes fitted forest thresholds and imputation statistics, but no training matrix, nearest-neighbor records, support vectors, original row-level outputs or identity mapping. A trained model is not a formal differential-privacy guarantee.
