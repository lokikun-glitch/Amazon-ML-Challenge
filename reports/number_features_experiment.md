# Number-aware D3 features: controlled ablation

Date: 2026-09-27. Config: D3 blocker, LR, clean 8K FIT / 2K TUNE / 10K VAL S1 protocol.
Raw numbers: `number_features_experiment.json`, `number_features_coef_check.json`, `address_numbers_extraction.json`.

## What changed

**Audit of the existing extraction** (`normalization.py`, left unchanged so the baseline stays frozen):

- **`postal_code`**: the last 5–6 digit run anywhere in the raw address.
  - The corpus contains no US ZIP codes; 0.00% of US addresses end in `ST 12345`.
  - It contains no standalone Indian PINs; about 1% carry a split `-624 103` form.
  - The field is therefore never a real postal code. For US it is a 5-digit house, unit or road number (98% equal to `street_number` when present), or a zero-padded house number.
- **`street_number`**: the first all-digit token of `address_norm`, where punctuation has already become whitespace. As a result:
  - `26/244` → `26` and `6-3-668/10/4` → `6`.
  - Leading zeros are kept, so `002798` ≠ `2798`.
  - `5527C` is skipped.

**New `address_numbers.py`** does label-aware, deterministic extraction:
- Fields: primary house number, head/atoms, compound, unit, postal code, number tokens.
- ZIPs are accepted only in `ST 12345` form at the end of the string.
- Compounds keep all their parts.
- Ordinals, sector, floor and road numbers are never promoted to house numbers.
- Ambiguous numbers only enter the token set.
- 52 regression tests.

**New pairwise features** (`build_number_features.py`) are added on top of the unchanged 28 existing features:

| Agreement (variants B and C) | Contradiction (variant B only) |
|---|---|
| `num_hn_head_equal`, `num_hn_full_equal`, `num_compound_equal`, `num_unit_equal`, `num_postal_equal`, `num_token_jaccard`, `num_token_overlap` | `num_hn_head_mismatch`, `num_compound_mismatch`, `num_unit_mismatch`, `num_postal_mismatch`, `num_token_contradiction`, `num_token_symdiff`, `num_strongname_contra` (name token-set ratio ≥ 0.90 AND a number contradiction; 0.90 is a fixed prior) |

Held fixed for every variant: candidate pairs, labels, split, LR hyperparameters, threshold grid, metric and multi-match rule.

## Results (VAL, 10K S1, each variant evaluated once)

| Criterion | A: existing (28) | B: +agree +contra (42) | C: +agree (35) | B−A | Pass? |
|---|---:|---:|---:|---:|---|
| Macro F0.5 | 0.7602 | **0.7945** | 0.7786 | +0.0343 | yes (≥ +0.003) |
| Precision | 0.8382 | 0.8966 | 0.8700 | +0.0584 | yes |
| Recall | 0.7628 | 0.7443 | 0.7545 | −0.0185 | trade-off (F0.5 weights precision) |
| Singleton FP % | 54.01 | 34.49 | 43.38 | −19.52 pp | yes (≥ 3 pp) |
| Matched S1 zero-pred % | 4.41 (416) | 5.45 (514) | 5.12 (483) | +1.04 pp | **no** (limit 1.0 pp; over by 0.04 pp) |
| US F0.5 | 0.7834 | 0.8174 | 0.8034 | +0.0340 | yes |
| India F0.5 | 0.7257 | 0.7604 | 0.7416 | +0.0347 | yes |
| Multi-match recall | 0.7501 | 0.7310 | 0.7414 | −0.0191 | flagged |
| Avg predictions/S1 | 3.095 | 2.831 | 2.960 | −0.264 | — |
| Duplicate claims | 5 | 3 | 8 | −2 | yes |
| TUNE-selected threshold z | 7.3 | 8.3 | 7.9 | | |

A reproduces the stored `decision_rules_D3.json` variant A exactly (the script asserts this).

Paired bootstrap over the same 10K VAL S1s (2,000 resamples):

| Comparison | Delta | 95% CI | US delta | India delta |
|---|---:|---|---:|---:|
| B − A | +0.0343 | [+0.0308, +0.0377] | +0.0340 [+0.0295, +0.0382] | +0.0347 [+0.0286, +0.0408] |
| B − C | +0.0159 | [+0.0135, +0.0184] | +0.0140 | +0.0187 |
| C − A | +0.0184 | [+0.0150, +0.0215] | +0.0200 | +0.0159 |

**Primary verdict: CONFIRMED IMPROVEMENT** (delta ≥ +0.003 and CI lower bound > 0, in both countries).

## Failure mode

These diagnostics were computed after all VAL results were fixed.

**Forensic subset** (training-side singletons whose A top candidate had z ≥ 7):

| Subset | Still predicted: A → B | FPs removed | FPs added |
|---|---|---|---|
| TUNE, 59 cases (out-of-sample) | 55 → 33 | 22 | 0 |
| TUNE, "number changed" category | 50 → 29 | | |
| FIT, 251 cases (in-sample, optimistic) | 237 → 155 | 84 | 2 |

**VAL pairs, A → B:**

| Pair type | FP pairs | TP pairs | Change |
|---|---|---|---|
| With a number contradiction | 3891 → 2130 | 2304 → 1845 | FP −45%, TP −20% |
| Without a contradiction | 1412 → 1193 | 23340 → 23140 | |

- The reduction is concentrated where numbers contradict.
- VAL singletons that A assigned a match: 310, of which 263 have a contradicting number in their top pair.
  - B clears 115 of the 263 contradiction cases but only 3 of the 47 without one.
- B also removes 1.3× as many contradiction FP pairs as C, which supports the specific contradiction hypothesis rather than "more number features".

The failure mode is reduced, not eliminated: 29 of 54 TUNE decoy cases are still predicted.

## Coefficients (B)

| Feature | Coef | | Feature | Coef |
|---|---:|---|---|---:|
| num_hn_head_equal | +0.254 | | num_hn_head_mismatch | +0.110 * |
| num_hn_full_equal | +0.049 | | num_compound_mismatch | −0.186 |
| num_compound_equal | +0.001 | | num_unit_mismatch | −0.109 |
| num_unit_equal | +0.086 | | num_postal_mismatch | −0.053 |
| num_postal_equal | −0.012 | | num_token_contradiction | −0.767 |
| num_token_jaccard | +0.463 | | num_token_symdiff | −0.463 |
| num_token_overlap | +0.804 | | num_strongname_contra | +0.069 * |

\* Unexpected positive signs, investigated on FIT only. Both are collinearity artifacts:

- **Co-occurrence:** P(token contradiction | head mismatch) = 0.94, and P(head mismatch | strongname_contra) = 0.91.
- **Univariate evidence:** head mismatch has a 0.045% positive rate vs 0.575% without it.
- **Net effect:** a strong-name pair flipping from agreeing to contradicted costs −1.62 logits in B.
- **Whole number block:** strong-name pairs with agreeing numbers get +3.78, and strong-name negatives with contradicted numbers get −0.62. That 4.4-logit gap compares with 2.1 in A.

Also collinear (correlation 0.92–0.95):
- the old `street_number_equal` (now −0.30) with `num_hn_head_equal`;
- `address_numeric_token_overlap` (−0.41) with `num_token_overlap`.

Individual coefficients should not be read in isolation.

## Cost

| Step | Cost |
|---|---|
| Extraction, once for 12.5M training records | 92 s (14 workers); parent peak 2.4 GB |
| Pair features per D3 split | about 95 s; peak 4.0 GB; 662 MB file (includes id columns kept for the alignment check) |
| LR fit | B 53 s, C 55 s (A was 43 s) |
