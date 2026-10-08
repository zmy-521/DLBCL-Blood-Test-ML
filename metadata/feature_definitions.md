# Predictor definitions

`data_dictionary.csv` contains the exact 40 Python column names, units, definitions and roles. `candidate_features.json` preserves original candidate ordering; `model/feature_order.json` is the exact 13-column inference order read from frozen model indices and checked against frozen metadata and the revision data packet. AGE and SEX describe cohorts and are not predictors.

Final preprocessing is a serialized development-fitted median imputer with keep_empty_features=True and scaling passthrough. Supply numbers in the dictionary units; no automatic unit conversion, clipping or outlier removal is performed. Numeric missing values are imputed; a missing required column is an error. Development has no missing predictor values. External has six case panels missing PCT, PDW, MPV and P-LCR (24 cells); none is a final RF13 predictor.

PCT means plateletcrit. Percentage features are expressed in percent, not proportions. A/G is dimensionless. BUN is retained as the source column label; urea-versus-urea-nitrogen terminology needs source confirmation before any reinterpretation or conversion. Case-source labels support the listed units; control units could not be independently verified from their original provenance. Algebraic QC does not establish independent assay harmonization.
