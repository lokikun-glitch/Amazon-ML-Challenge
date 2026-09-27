#!/bin/sh
# Shard smoke test: same 1,000 S1/country sample as the accepted production smoke run.
set -e
cd "$(dirname "$0")"
R=../../../output
for RUN in shard_smoke shard_smoke_rerun; do
  for C in FR IN US; do
    python run_country_shard.py --country $C --smoke-per-country 1000 --out-root $R/$RUN 2> $R/$RUN.$C.log
  done
  python consolidate_shards.py --smoke --shard-root $R/$RUN --out-dir $R/$RUN > $R/$RUN.consolidate.json
  python check_outputs.py --smoke --dir $R/$RUN > /dev/null
done
echo SHARD_SMOKE_DONE
