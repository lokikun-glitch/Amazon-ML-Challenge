# Business Entity Resolution Challenge
## Engineering Problem Statement for Claude

You are helping build a competitive, reproducible ML/data-science solution for a **Business Entity Resolution Challenge**.

Your job is to understand the complete task, inspect the supplied dataset, design the pipeline, implement it, validate it, and iteratively improve it using only the allowed data.

---

# 1. Core Problem

We have business records from **three independent data sources**:

- Source 1: deduplicated reference/master source
- Source 2: noisy business records
- Source 3: noisy business records

There are no common identifiers across sources.

For **every Source 1 entity**, determine **all matching records from Source 2 and Source 3** that refer to the same real-world business.

A Source 1 entity can have:

- zero matches
- one match
- multiple matches

Source 1 is the reference source.

The task is a large-scale **entity resolution / record linkage** problem.

---

# 2. Data Files

Training:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

Test:

```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

All files are **tab-separated TSV files**.

Always read them with:

```python
pd.read_csv(path, sep="\t")
```

Do NOT assume commas are separators.

---

# 3. Columns

Each source file contains:

```text
entity_id
business_name
business_address
country
```

## entity_id

Unique record identifier.

Prefix indicates source:

```text
S1-...
S2-...
S3-...
```

There is no separate `source` column.

The source is determined by the file and/or entity ID prefix.

## business_name

May contain:

- abbreviations
- legal suffixes
- typos
- transliterations
- DBA/trade names
- punctuation differences
- `&` vs `and`
- word-order changes

## business_address

May contain:

- abbreviations
- partial addresses
- missing components
- transliteration variants
- landmark-based references
- municipal numbering differences
- component reordering

## country

Training contains:

```text
US
India
```

The test set additionally contains:

```text
France
```

Treat country as an **open-set string field**.

Do NOT hard-code the pipeline to only US and India.

Every test Source 1 entity, including France, must appear in the output.

---

# 4. Ground Truth

Training ground truth:

```text
dataset/train/train_ground_truth.tsv
```

Columns:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list of matching Source 2 and/or Source 3 IDs.

An empty value means the Source 1 entity has no matches.

Example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

The training ground truth should be used to build supervised training/validation data.

---

# 5. Required Pipeline

Build a complete pipeline:

```text
RAW DATA
   ↓
DATA INSPECTION
   ↓
NORMALIZATION
   ↓
BLOCKING / CANDIDATE GENERATION
   ↓
candidate_pairs.tsv
   ↓
PAIR FEATURE ENGINEERING
   ↓
MATCHING MODEL
   ↓
MATCH / NO-MATCH DECISION
   ↓
matching_results.tsv
   ↓
VALIDATION
```

The pipeline must be reproducible end-to-end.

---

# 6. CRITICAL: Blocking / Candidate Generation

Blocking is one of the most important parts of the challenge.

The challenge is designed for very large-scale data. Comparing every Source 1 record against every Source 2 and Source 3 record is unacceptable.

The blocking stage must dramatically reduce the search space.

Conceptually:

```text
S1 entity
   ↓
blocking
   ↓
small candidate set
   ↓
expensive matching model
```

The organizers explicitly evaluate the candidate-generation approach.

The final `candidate_pairs.tsv` must contain the **exact candidate set actually passed to the final matching model during inference**.

If multiple blocking/filtering stages exist:

```text
raw records
   ↓
block 1
   ↓
block 2
   ↓
block 3
   ↓
final candidates
   ↓
ML model
```

then `candidate_pairs.tsv` must contain the candidates immediately before the ML model.

It must NOT contain candidates from an earlier blocking stage that were subsequently discarded.

Every final predicted match must exist in `candidate_pairs.tsv`.

---

# 7. Blocking Objective

Blocking has two competing goals:

### Goal A: High recall

A true match must enter the candidate set.

If the true match is removed during blocking:

```text
ML model cannot recover it.
```

Therefore blocking determines the upper bound of achievable recall.

### Goal B: Small candidate sets

Do not pass thousands of irrelevant records into the expensive matcher if they can be eliminated cheaply.

The challenge specifically values a smaller candidate set per Source 1 entity.

Therefore optimize:

```text
high candidate recall
+
low candidate count
```

Measure both quantitatively.

Useful blocking ideas to investigate include:

- normalized exact name blocks
- name token blocks
- character n-gram blocks
- country + name prefix
- country + distinctive name token
- postal/PIN code blocks when available
- address token blocks
- street-number blocks
- TF-IDF retrieval
- character TF-IDF retrieval
- multiple blocking channels whose candidates are unioned
- subsequent cheap candidate pruning

Do not blindly implement all of these. Inspect the actual data and measure their recall and candidate reduction before deciding.

---

# 8. Normalization

Create robust normalized representations for names and addresses.

Do NOT rely on a single aggressively normalized string.

Potential transformations to investigate:

- lowercase
- Unicode normalization
- punctuation normalization
- whitespace normalization
- `&` ↔ `and`
- common legal suffix normalization
- common business abbreviations
- address abbreviation normalization
- tokenization
- numeric token extraction
- postal/PIN extraction
- character n-grams
- transliteration-aware representations where useful

Preserve multiple representations when useful.

Example:

```text
ABC Pvt. Ltd.
```

could produce representations such as:

```text
abc pvt ltd
abc private limited
abc
```

Do not remove information that could help distinguish businesses.

The exact normalization strategy must be driven by the actual dataset.

---

# 9. Matching Model

After blocking, each pair should be treated as a binary matching problem:

```text
(S1 record, candidate S2/S3 record)
        ↓
features
        ↓
match probability / score
        ↓
MATCH or NO MATCH
```

The model should use the training ground truth.

Potential pair features include:

## Name features

- exact normalized-name match
- token Jaccard similarity
- token overlap
- character n-gram similarity
- Levenshtein/edit similarity
- Jaro/Jaro-Winkler if useful
- TF-IDF cosine similarity
- name length difference
- token count difference
- acronym similarity
- prefix/suffix similarity

## Address features

- normalized address similarity
- token Jaccard
- character similarity
- TF-IDF cosine
- numeric token overlap
- street-number match
- postal/PIN match
- city/state token overlap
- address length difference
- distinctive-token overlap

## Cross-field features

- country equality
- country compatibility
- name similarity × address similarity
- strong-name + weak-address patterns
- weak-name + strong-address patterns
- numeric/address evidence
- missing-field indicators

Do not assume all features are useful. Perform validation and feature analysis.

---

# 10. Model Choice

A huge language model is NOT required.

Strong classical ML / tabular approaches should be considered first.

Potential models:

- Logistic Regression
- Random Forest
- ExtraTrees
- XGBoost
- LightGBM
- CatBoost
- other appropriate licensed gradient-boosting models

A transformer/LLM may be considered only if justified by validation results and licensing/parameter constraints.

The final model must comply with:

```text
MIT or Apache 2.0 license
≤ 8 billion parameters
```

Do not use a model whose license violates the competition rules.

---

# 11. Precision-Heavy Evaluation

The leaderboard metric is:

```text
F0.5
```

Formula:

```text
F0.5 = (1.25 × Precision × Recall) /
       (0.25 × Precision + Recall)
```

It is calculated **per Source 1 entity**, then macro-averaged across Source 1 entities.

Precision is weighted more heavily than recall.

Therefore false merges are especially harmful.

Do NOT simply select the highest-scoring candidate for every Source 1.

The model needs an explicit:

```text
MATCH
or
NO MATCH
```

decision.

---

# 12. Singletons

Singletons are very important.

If a Source 1 entity truly has no Source 2/3 matches:

```text
S1-X → empty
```

and the model correctly predicts empty:

```text
F0.5 = 1.0
```

for that entity.

If the model incorrectly predicts any match:

```text
F0.5 = 0.0
```

for that entity.

Therefore the system must be conservative enough to correctly identify genuine singletons.

Do not force every Source 1 entity to have a match.

---

# 13. Validation Strategy

There is no ground truth for the test set.

Therefore create a validation split from the training data.

IMPORTANT:

Split by **Source 1 entity**, not by individual candidate pairs.

For example:

```text
Training S1 entities
Validation S1 entities
```

Do not allow records belonging to the same Source 1 entity to leak across train and validation.

For validation:

```text
training S1 entities
       ↓
blocking
       ↓
candidate pairs
       ↓
train matching model
       ↓
validation S1 entities
       ↓
blocking
       ↓
matching
       ↓
predictions
       ↓
macro F0.5
```

Track at minimum:

```text
F0.5
Precision
Recall
True positive count
False positive count
False negative count
Singleton accuracy
Blocking recall
Average candidate count
Median candidate count
Candidate count distribution
Reduction ratio
```

---

# 14. Avoid Data Leakage

Be careful not to leak validation information.

Examples of leakage to avoid:

- fitting TF-IDF on validation/test jointly when the validation experiment is meant to simulate unseen data
- training the classifier on validation pairs
- using validation ground truth to construct blocking rules
- using test ground truth, which does not exist
- tuning thresholds directly on the final test set

All model/tuning decisions should be made using training data and held-out validation.

---

# 15. Test Set Requirements

For:

```text
dataset/test/test_source1.tsv
```

EVERY Source 1 entity must receive exactly one output row.

Even if no candidate exists.

If there are no matches:

```text
source1_entity_id    matched_entity_ids
S1-xxxxx
```

with an empty second field.

Never omit a Source 1 entity.

---

# 16. matching_results.tsv

Required columns:

```text
source1_entity_id
matched_entity_ids
```

Rules:

1. Exactly one row per test Source 1 entity.
2. Every Source 1 entity must appear.
3. `matched_entity_ids` contains comma-separated S2/S3 IDs.
4. Empty when no match exists.
5. No duplicate IDs within one list.
6. IDs must exist in the test Source 2/3 data.
7. Never output an S1 ID as a match.
8. Every predicted match must also occur in `candidate_pairs.tsv`.

Example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

---

# 17. candidate_pairs.tsv

Required columns:

```text
source1_entity_id
candidate_entity_ids
```

Rules:

1. Exactly one row per Source 1 entity.
2. Candidate IDs may only come from S2/S3.
3. No duplicates.
4. Empty if blocking produces no candidates.
5. This must represent the exact final candidate set passed to the matching model.
6. Every final predicted match must be present here.

Example:

```text
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

---

# 18. Submission Validation

The challenge provides:

```text
utils/validate_submission.py
```

Run:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It should return:

```text
PASS
```

before submission.

Do not assume the output is valid without running the validator.

---

# 19. External Data Is STRICTLY PROHIBITED

Do NOT use:

- Google
- Google Maps
- business directories
- government business registries
- commercial entity-resolution APIs
- geocoding APIs
- external company databases
- internet-based business lookup
- external data augmentation
- any external service that resolves or enriches business identities

Do not use external information to determine whether two businesses are the same.

The solution must rely only on:

```text
provided challenge data
+
algorithms
+
allowed ML models/libraries
```

Any evidence of external entity lookup can result in disqualification.

---

# 20. Important Engineering Principle

Do not jump directly into writing the final model.

First inspect the dataset.

Perform data forensics:

```text
1. Number of S1 records
2. Number of S2 records
3. Number of S3 records
4. Number of ground-truth links
5. Distribution of matches per S1
6. Singleton percentage
7. Country distribution
8. Missing-value rates
9. Name length/token statistics
10. Address length/token statistics
11. Exact normalized-name overlap
12. Exact normalized-address overlap
13. Name/address duplication patterns
14. Cross-source similarities
15. Hard negative patterns
```

Then design the blocking strategy based on evidence.

---

# 21. Blocking Evaluation

For every candidate-generation method, measure:

```text
True Match Recall:
How many true matches survive blocking?

Average Candidates:
Average number of candidates per S1.

Median Candidates

95th percentile Candidates

Maximum Candidates

Reduction Ratio:
1 - (# candidates generated / # possible pairs)
```

The most useful blocker is not necessarily the one with the highest raw recall.

The goal is:

```text
very high recall
+
small candidate set
```

Use multiple blocking channels if that provides a better trade-off.

---

# 22. Suggested Experimental Baseline

Start with a simple baseline:

```text
Normalization
    ↓
country equality
    ↓
normalized name similarity
    ↓
normalized address similarity
    ↓
simple classifier
    ↓
threshold
```

Then progressively improve.

Do NOT build a giant complicated pipeline before measuring the baseline.

Keep experiment results.

Example experiment table:

```text
Experiment | Blocking Recall | Avg Candidates | Precision | Recall | F0.5
-----------|------------------|-----------------|-----------|--------|-----
Baseline   |                  |                 |           |        |
Block A    |                  |                 |           |        |
Block B    |                  |                 |           |        |
Model A    |                  |                 |           |        |
Model B    |                  |                 |           |        |
```

---

# 23. Candidate Set Design

Think of blocking as a retrieval system.

A strong design may look like:

```text
                  S1
                   │
       ┌───────────┼────────────┐
       ↓           ↓            ↓
 Name Block   Address Block   TF-IDF Block
       │           │            │
       └───────────┼────────────┘
                   ↓
                 UNION
                   ↓
            cheap pruning
                   ↓
            final candidates
                   ↓
             ML matcher
```

Measure every stage.

Do not accidentally make the final candidate set huge just to increase recall.

---

# 24. Matching Decision

The final model may output:

```text
P(match)
```

Do not automatically use:

```text
P > 0.5
```

Tune the threshold on held-out validation according to macro F0.5.

Potentially investigate different confidence policies for:

- very strong exact matches
- ambiguous name matches
- strong address matches
- sparse records
- singletons
- multiple high-confidence candidates

But avoid overfitting complicated rules unless validation supports them.

---

# 25. Multiple Matches

A Source 1 business may legitimately map to multiple Source 2/3 records.

Therefore do not assume:

```text
one S1 → one S2/S3
```

Instead:

```text
one S1 → zero, one, or many
```

The model should independently evaluate candidate pairs, followed by appropriate thresholding/post-processing.

Use training ground truth to understand how often multi-match cases occur and whether there are source-specific patterns.

---

# 26. Output Ordering

Unless the validator specifies another requirement, keep output deterministic.

Use stable ordering for:

- Source 1 IDs
- matched IDs
- candidate IDs

Avoid nondeterministic set iteration.

The same input should generate the same output.

---

# 27. Reproducibility

The final repository must contain:

```text
code/
└── business_entity_resolution/
    ├── src/
    ├── README.md
    └── requirements.txt
```

The README must explain exactly how to:

```text
1. Load the data
2. Train the model
3. Run blocking
4. Generate candidates
5. Train/infer matching
6. Generate outputs
7. Validate outputs
```

Pin dependencies where practical.

The pipeline should regenerate:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

from the supplied data.

---

# 28. Final Submission Package

Required structure:

```text
<team_name>_submission.zip
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

The methodology document must describe:

- methodology
- candidate-generation/blocking strategy
- model architecture
- feature engineering
- relevant implementation details

---

# 29. Recommended Work Order

Follow this order.

## Phase 1: Dataset Forensics

Inspect all files.

Determine:

```text
dataset sizes
columns
missingness
country distribution
match distribution
singleton rate
name/address characteristics
```

## Phase 2: Ground Truth Analysis

Study:

```text
match counts
S2/S3 source proportions
name variation
address variation
hard positives
hard negatives
```

## Phase 3: Normalization

Build reusable name/address normalization functions.

Test them on real examples.

## Phase 4: Blocking

Implement several candidate-generation strategies.

Measure:

```text
blocking recall
candidate count
reduction ratio
```

Select a strong high-recall/low-candidate approach.

## Phase 5: Pair Features

Build name/address/country similarity features.

## Phase 6: Baseline Matcher

Train a simple classifier.

## Phase 7: Threshold Optimization

Optimize macro F0.5 on held-out validation.

## Phase 8: Error Analysis

Inspect:

```text
false positives
false negatives
singleton errors
high-candidate entities
low-confidence matches
```

## Phase 9: Iterate

Improve only where validation evidence shows a problem.

## Phase 10: Final Test Inference

Run the frozen pipeline on:

```text
test_source1
test_source2
test_source3
```

Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

## Phase 11: Validate

Run:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Require:

```text
PASS
```

## Phase 12: Package

Create the final ZIP with code, outputs, README, requirements, and methodology.

---

# 30. What Claude Should NOT Do

Do NOT:

- use external business lookup
- use Google/Maps/geocoding
- hard-code US/India
- force one match per S1
- ignore singleton cases
- compare every record against every record
- produce candidate_pairs.tsv from an early blocking stage
- output matches that are absent from candidate_pairs.tsv
- train on validation data
- tune using the hidden test set
- assume exact names are the only useful signal
- assume addresses are always complete
- blindly use an LLM because the task is called ML
- sacrifice candidate-set size unnecessarily
- optimize only pairwise accuracy instead of macro F0.5
- ignore reproducibility
- submit without running the validator

---

# 31. Primary Optimization Target

The final system should balance:

```text
                    ┌───────────────────┐
                    │   HIGH F0.5       │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        High precision    Good recall    Singleton accuracy
              │
              │
       ┌──────┴──────┐
       ↓             ↓
Strong matching   Good blocking
                  │
                  ↓
          Small candidate set
```

Remember:

```text
BLOCKING
    = maximize true-match retention while minimizing candidates

MATCHING
    = maximize precision/recall trade-off for macro F0.5

DECISION
    = be conservative enough to avoid false merges and correctly identify singletons
```

---

# 32. Expected Deliverables from Claude

At the end of the work, produce:

### A. Working pipeline

Runnable from the supplied dataset.

### B. `matching_results.tsv`

Valid final predictions.

### C. `candidate_pairs.tsv`

Exact final candidate set used by the matching model.

### D. Validation report

Include:

```text
F0.5
Precision
Recall
Singleton performance
Blocking recall
Average candidates
Median candidates
95th percentile candidates
Reduction ratio
```

### E. Error analysis

Explain the major remaining:

```text
false positives
false negatives
singleton mistakes
blocking misses
```

### F. Reproducible code

Under:

```text
code/business_entity_resolution/src/
```

### G. README

Exact reproduction instructions.

### H. Requirements

Pinned dependencies.

### I. Methodology document

Explain:

```text
normalization
blocking
candidate generation
features
model
thresholding
validation
limitations
```

---

# 33. Final Instruction to Claude

Treat this as a **real competitive entity-resolution engineering task**, not a toy string-matching exercise.

Before implementing sophisticated methods:

1. Inspect the actual dataset.
2. Quantify the problem.
3. Establish a reproducible baseline.
4. Measure blocking recall and candidate reduction.
5. Build supervised pair features from ground truth.
6. Train and validate a precision-aware matcher.
7. Optimize macro F0.5.
8. Perform error analysis.
9. Improve the weakest components.
10. Generate valid deterministic outputs.
11. Run the official validator.
12. Keep the entire solution reproducible and compliant with the no-external-data rule.

Do not make assumptions about the data when they can be measured directly.

When making an engineering decision, prefer:

```text
measurement → experiment → validation → decision
```

over intuition alone.

The final objective is not merely to produce matches. It is to produce a **high-quality, high-precision, high-recall, scalable, reproducible entity-resolution system with an efficient candidate-generation stage that satisfies every competition constraint**.
