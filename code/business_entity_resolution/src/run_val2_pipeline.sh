#!/bin/sh
# Second untouched validation sample (val2): candidates + features for D3 and C.
set -e
cd "$(dirname "$0")"
R=../../../reports
for CFG in D3 C; do
  python generate_dev_candidates.py --config $CFG --val2-seed 2027 2> $R/gen_candidates_${CFG}_val2.log
  python build_pairwise_features.py --config $CFG --split val2 2> $R/features_${CFG}_val2.log
done
echo VAL2_PIPELINE_DONE
