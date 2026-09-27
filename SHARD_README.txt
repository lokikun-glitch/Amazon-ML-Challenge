FINAL-B-H2 country shard kit  (Python 3.13.5; packages pinned in shard_requirements.txt)

1. Unzip anywhere, e.g. C:\shard  (keep the folder structure).
2. python -m pip install -r shard_requirements.txt
3. cd code\business_entity_resolution\src
4. python run_country_shard.py --country IN      (or US, or FR)
   - PYTHONHASHSEED=0 is set automatically (the script relaunches itself).
   - Needs ~11 GB free RAM and ~15 GB free disk. Runtime: IN ~6.6 h, US ~4.5 h, FR ~0.9 h.
   - Progress: printed every 2,000 S1 (S1 done, pairs, zero-candidate %, predictions, H2).
5. When it prints "status=success", copy the whole folder
       output\shards\<CODE>\     (candidate_pairs_<CODE>.tsv, matching_results_<CODE>.tsv, manifest_<CODE>.json)
   back to the main PC into  <project>\output\shards\<CODE>\
Do not edit or re-save the .tsv files (their SHA-256 is checked in the manifest).
