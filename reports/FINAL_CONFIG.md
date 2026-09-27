# Final frozen configuration: FINAL-B-H2

Frozen 2026-09-27 after the VAL2 replication gate. Details are in `val2_B_H2_evaluation.json`.
Any change to this configuration requires a newly authorized experiment.

## Pipeline

1. **Normalization:** `normalization.py` (`build_normalized_frame`), unchanged.
2. **Blocking: D3**, as defined in `generate_dev_candidates.CONFIGS["D3"]`. The country value is part of every blocking key.

   | Channel | Rule |
   |---|---|
   | B1 | exact `name_norm` |
   | B2 | exact `name_no_suffix` |
   | B3 | name tokens, max_df = 500, K = 4 |
   | B5 | address tokens, max_df = 2000, K = 6 |
   | B7a | exact `postal_code` as implemented; for US this is effectively a house number |

3. **Features (42):**
   - 28 base features from `build_pairwise_features.py`.
   - 14 number features from `build_number_features.py` (`AGREEMENT` + `CONTRADICTION`), using the extractor `address_numbers.extract`.
   - Frozen known limitation: a number glued to the following word (for example `2/1Chattaarpur`) is not split. The expected-to-fail test in `tests/test_address_numbers.py` records this.
4. **Model:** StandardScaler + LogisticRegression (class_weight = "balanced", C = 1.0, max_iter = 1000), fit on the 8K FIT S1 only.
   - The persisted model is `data_cache/models/B_frozen.npz`, containing the feature order, scaler mean and scale, coefficients, intercept, `threshold_z` and `h2_t_low`.
   - It reproduces the stored original-VAL scores exactly.
5. **Decision, applied per S1:**
   - **B rule:** predict every candidate with z ≥ 8.3 (p ≥ 0.99975). Multiple matches are kept.
   - **H2:** only if the S1 has no candidate with z ≥ 8.3, predict its single top-scoring candidate when all of the following hold:
     - z ≥ 3.0
     - `num_hn_head_equal` == 1
     - `num_hn_head_mismatch` == 0
     - `num_compound_mismatch` == 0
     - `num_token_contradiction` == 0

     Otherwise the S1 is predicted empty. H2 never adds candidates to an S1 that already has a prediction.
   - Parameters: 8.3 was selected on TUNE (global grid); 3.0 and the agreement condition were selected on TUNE (H2 analysis).

## Evidence

| Configuration | Original VAL (10K) | VAL2 (10K) | Decision |
|---|---:|---:|---|
| A (28 features, z ≥ 7.3) | 0.7602 | 0.7595 | superseded |
| B (42 features, z ≥ 8.3) | 0.7945 | 0.7952 | superseded by B + H2 |
| **B + H2** | not evaluated (by design; original VAL is frozen) | **0.8009** | **FINAL** |

- **H2 − B on VAL2:** +0.0057, 95% CI [+0.0039, +0.0074], P(Δ ≤ 0) = 0.
  - US: +0.0053 [+0.0032, +0.0075].
  - India: +0.0062 [+0.0035, +0.0091].
- **B − A on VAL2:** +0.0357 [+0.0323, +0.0391]. This independently replicates the original-VAL result of +0.0343.

## Known trade-offs of the final configuration (VAL2)

- **Singleton FP rises 3.11 pp** (38.51% → 41.62%). This is the price of H2's recovery of zero-prediction S1s.
- **Precision drops 0.29 pp**, which is under the 1 pp guardrail.
- **Matched-S1 zero predictions fall 1.14 pp** (5.13% → 3.99%). This recovers the zero-prediction guardrail that was flagged for B relative to A.
- **Multi-match recall is 0.735**, below A's 0.750 on the original VAL. That is B's precision/recall trade-off and remains.
- **France is untested:** it is absent from all training and validation data, and is 15% of test S1s.
