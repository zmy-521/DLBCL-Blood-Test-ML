# Controlled-access input contract

No inputs described in this document are shipped, except the explicitly synthetic inference CSV.

## Preferred exact-analysis route
Supply these files in `run/data_v2/` from the authorized frozen analysis:

- `development_grouped_anonymous.csv`
- `development_strict_baseline_anonymous.csv`
- `external_grouped_anonymous.csv`
- `external_strict_baseline_anonymous.csv`

Columns: `center` (development/external), `group_key` (stable anonymous person grouping), `group` (0 healthy control, 1 DLBCL), `episode_id` (anonymous episode), `AGE` (years), `SEX` (M/F/empty), and all 40 numeric columns in `candidate_features.json`. The public adapter renames the original anonymous grouping field to `group_key` and encodes sex as M/F. Preserve grouping values and row order for exact fold reproduction; changing either can change group-stratified split allocation even with the same seed. Preserve numeric values and missingness. Median filling occurs during model fitting, not in these input files. Baseline rows must be a true subset of primary records and contain one episode per person. Controls occur once. Centers must not overlap.

## Optional normalized-source route
This is an interface adaptation of the frozen pairing logic, not a replacement for hospital-specific extraction or clinical eligibility review. Input templates containing private records must remain local. Provide:

1. `local_inputs/approved_panels.csv`: one reconstructed chemistry or CBC source panel per row. Columns `center`, `group_key`, `panel_key`, `panel_type` (chemistry/cbc), `sampling_times`, `test_times` (pipe-separated parseable timestamps or empty), `age_source` (numeric age or pipe-separated values), `sex` (M/F/empty), `included` and `identity_uncertain` (0/1), plus the 40 candidate columns, blank for inapplicable panel type. Panels must be reconstructed by verified item labels, not unverified column positions. Multiple distinct values for an analyte within one source panel become missing pending source review, as in the original extraction. All selected source panels, including unpaired panels, must be supplied for strict chronology assessment.
2. `local_inputs/approved_controls.csv`: `center`, `group_key`, `AGE`, `SEX`, all 40 predictors, one row per investigator-confirmed distinct control, no cross-center or case/control overlap.

Run `python code/preprocess.py`. `build_episodes.py` retains the original one-chemistry/one-CBC same-day pairing and strict chronology implementation. `preprocess.py` adapts the input interface and control preparation. `identity_linkage.py` exposes the original nonconflicting-component rule using generic local keys; custodian verification and uncertain-identity exclusions are mandatory. `stable_group` makes stable HMAC tokens if a fixed local secret and verified entity are supplied. Changing tokens may change CV split ordering. Never publish the secret or linking inputs.

The normalized adapter writes confidential intermediary files under ignored `run/preprocessing_private/`. It has synthetic unit checks but was not rerun on original hospital exports for this release. Exact regeneration of source eligibility and identity components requires the controlled original extraction evidence; it cannot be independently reconstructed from the public files. Do not imply that unrestricted raw-to-results replication is possible without authorized data access.
