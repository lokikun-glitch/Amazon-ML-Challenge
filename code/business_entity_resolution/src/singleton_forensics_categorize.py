"""
Step 2 of the singleton forensics: characterize HOW record pairs differ
(house/unit numbers, added name tokens) for accepted true positives (A_acc),
high-scoring singleton FPs (C), matched-S1 FPs (D) and random negatives (B),
then assign each C case a failure-mode category.

Categories are rule-assigned from observable record differences and then
checked by manual reading of all C cases (see singleton_forensics_cases.tsv).
GT is used only for group membership and the "claimed by another S1" check.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
CACHE_DIR = os.path.join(ROOT, "data_cache", "normalized")
REPORT_DIR = os.path.join(ROOT, "reports")

# legal-form / filler tokens: differences in these are ordinary record noise
LEGAL = {"ltd", "limited", "llc", "l", "c", "inc", "incorporated", "corp", "corporation", "co", "company",
         "pvt", "private", "public", "llp", "p", "pc", "plc", "pllc", "lp", "the", "m", "s", "ms", "mr", "smt",
         "sri", "shri", "and", "a", "of"}
PIN_RE = re.compile(r"^[1-9]\d{5}$")   # Indian PIN codes (never start with 0) are not house numbers;
#                                        zero-padded US numbers like "002798" must survive
# NB: US `postal_code` in the normalized cache is NOT a ZIP -- for 98% of the 11% of US
# records that have one it equals the 5-digit house number -- so it is not excluded here.

# Manual reading of all 310 C cases: rule category overridden where the records say otherwise.


def load_recs(ids, files):
    out = []
    for f in files:
        t = pq.read_table(os.path.join(CACHE_DIR, f), columns=["entity_id", "name_norm", "address_norm", "postal_code"])
        out.append(t.filter(pc.is_in(t["entity_id"], value_set=pa.array(sorted(ids)))).to_pandas())
    return pd.concat(out).set_index("entity_id")


def nums(addr, postal):
    toks = addr.split() if addr else []
    return {t for t in toks if any(ch.isdigit() for ch in t) and not PIN_RE.match(t)}


def core(name):
    return {t for t in name.split() if t not in LEGAL} if name else set()


def describe(s_name, c_name, s_addr, c_addr, s_pc, c_pc):
    ns, nc = nums(s_addr, s_pc), nums(c_addr, c_pc)
    cs, cc = core(s_name), core(c_name)
    if not c_addr:
        num = "cand_addr_missing"
    elif ns and nc:
        num = "numbers_equal" if ns == nc else ("numbers_cand_superset" if ns < nc else
                                                 ("numbers_cand_subset" if nc < ns else "numbers_differ"))
    elif not ns and nc:
        num = "number_added_by_cand"
    elif ns and not nc:
        num = "number_dropped_by_cand"
    else:
        num = "no_numbers"
    extra = cc - cs
    missing = cs - cc
    if cs == cc:
        nm = "core_equal"
    elif not missing and extra:
        nm = "cand_adds_token"
    elif missing and not extra:
        nm = "cand_drops_token"
    else:
        nm = "token_substituted"
    return num, nm, sorted(extra), sorted(missing)


NUM_CHANGED = {"numbers_differ", "number_added_by_cand", "numbers_cand_subset"}

# Manual reading of all 310 C cases: where the records say something the rule cannot see.
MANUAL_OVERRIDES = {
    # same street + same number, name differs only by a 1-2 char edit, no decoy token:
    # indistinguishable from real-match noise -> possible GT omission (NOT confirmed)
    "S1-620162779": ("possible_gt_omission", "AO Herbs vs AE Herbs, identical address"),
    "S1-741413560": ("possible_gt_omission", "Brinna T. Gonzalez CPA vs BPA, same number/street"),
    # names share only a generic token ('capital') at an identical address
    "S1-105764270": ("shared_address", "Nguyen Capital vs Thomas Capital, identical address"),
    # decoy-style number perturbations the digit-set rule misses (suffix letters / zero padding)
    "S1-437245617": ("near_duplicate_number_changed", "27412 vs 27415A"),
    "S1-151183277": ("near_duplicate_number_changed", "12815 vs 12817d"),
    "S1-133949477": ("near_duplicate_number_changed", "25307 vs 0025320"),
    # character-obfuscated name (0 for o, accent) so no exact core token is shared -> not a shared address
    "S1-1738123": ("near_duplicate_number_changed", "Macdonald & Schlabach vs Macd0nald & Schlabach (accented), 351 vs 356"),
}


def categorize():
    """Assign one primary failure mode per C case (rules + MANUAL_OVERRIDES), plus secondary flags."""
    d = pd.read_parquet(os.path.join(REPORT_DIR, "singleton_forensics_diffs.parquet"))
    d = d[d["group"] == "C"].drop(columns=["group", "z"])
    c = pd.read_csv(os.path.join(REPORT_DIR, "singleton_forensics_cases.tsv"), sep="	", dtype=str,
                    keep_default_na=False)
    derived = ["num_pattern", "name_pattern", "extra", "missing", "category", "category_reason"]
    c = c.drop(columns=[x for x in c.columns if x in derived or x.startswith("flag_")])  # idempotent re-runs
    m = c.merge(d, on=["s1_id", "cand_id"], how="left")
    assert m["num_pattern"].notna().all()
    cats, why = [], []
    for r in m.itertuples():
        overlap = len(core(r.s1_name_norm) & core(r.cand_name_norm))
        if r.s1_id in MANUAL_OVERRIDES:
            k, w = MANUAL_OVERRIDES[r.s1_id]
            w = "manual: " + w
        elif r.num_pattern == "cand_addr_missing":
            k, w = "other_same_name_cand_address_missing", "candidate has no address; GT links it to a different-location S1"
        elif overlap == 0:
            k, w = "shared_address", "unrelated business name at the same address"
        elif r.num_pattern in NUM_CHANGED:
            k, w = "near_duplicate_number_changed", f"house/unit number differs; name {r.name_pattern}"
        elif r.name_pattern in ("cand_adds_token", "token_substituted"):
            k, w = "near_duplicate_name_token_numbers_not_contradicted", f"no house number contradicted; name {r.name_pattern} (+{r.extra})"
        else:
            k, w = "possible_gt_omission", "same numbers, core name equal"
        cats.append(k)
        why.append(w)
    m["category"] = cats
    m["category_reason"] = why
    # secondary (non-exclusive) flags from observable rarity / script
    m["flag_common_name"] = (m["cand_name_freq"].astype(float) >= 5) | (m["s1_name_freq_pool"].astype(float) >= 5)
    m["flag_common_address"] = m["cand_addr_freq"].replace("", "0").astype(float) >= 5
    m["flag_nonlatin_cand_name"] = m["cand_name_norm"].map(lambda s: any(ord(ch) > 0x24F for ch in s))
    m["flag_name_adds_decoy_token"] = m["name_pattern"].isin(["cand_adds_token", "token_substituted"])
    m["flag_gt_claimed_by_other_s1"] = m["gt_claimed_by_other_s1"].astype(int) > 0
    m.to_csv(os.path.join(REPORT_DIR, "singleton_forensics_cases.tsv"), sep="	", index=False)
    n = len(m)
    summ = {"n_cases": n, "primary_category": {}, "secondary_flags": {}, "by_z": {}}
    for k, v in m["category"].value_counts().items():
        summ["primary_category"][k] = {"n": int(v), "pct": round(100 * v / n, 1)}
    for f in [c for c in m.columns if c.startswith("flag_")]:
        summ["secondary_flags"][f] = {"n": int(m[f].sum()), "pct": round(100 * m[f].mean(), 1)}
    z = m["z"].astype(float)
    for zl in (7, 9, 11):
        summ["by_z"][f"z>={zl}"] = m.loc[z >= zl, "category"].value_counts().to_dict()
    summ["by_country"] = m.groupby("country")["category"].value_counts().unstack(fill_value=0).to_dict("index")
    summ["not_observed"] = {
        "transliteration_collision": "0 as primary cause: of the 3 non-Latin candidate names, 2 (Gujarati/Hindi) scored high "
                                     "on a shared address and 1 (mixed Tamil/Latin) is a number-changed decoy",
        "address_number_collision": "0: no case where an equal house number on a DIFFERENT street drove the score",
        "blocking_artifact": "0: every case was retrieved by a name-token (b3) and/or address-token (b5) channel "
                             "and scores high on genuine name+address token overlap",
        "generic_common_name": "not a primary cause: see flag_common_name; names in C are RARER than in A_acc",
    }
    with open(os.path.join(REPORT_DIR, "singleton_forensics_categories.json"), "w") as f:
        json.dump(summ, f, indent=2, default=int)
    print(json.dumps(summ, indent=1, default=int))


def main():
    if "--categorize-only" in __import__("sys").argv:
        return categorize()
    P = pd.read_parquet(os.path.join(REPORT_DIR, "singleton_forensics_pairs.parquet"))
    rng = np.random.default_rng(0)
    sel = P[P["g_C"] | P["g_D"] | P["g_A_acc"] | P["g_B"]].copy()
    # keep A_acc and B at a manageable size (uniform subsample)
    for g, n in (("g_A_acc", 8000), ("g_B", 4000)):
        idx = sel.index[sel[g] & ~sel["g_C"] & ~sel["g_D"]]
        drop = idx[rng.permutation(len(idx))[n:]] if len(idx) > n else []
        sel = sel.drop(drop)
    s1 = load_recs(set(sel["s1_id"]), ["train_source1.parquet"])
    cd = load_recs(set(sel["cand_id"]), ["train_source2.parquet", "train_source3.parquet"])
    rows = []
    for r in sel.itertuples():
        a, b = s1.loc[r.s1_id], cd.loc[r.cand_id]
        num, nm, extra, missing = describe(a.name_norm, b.name_norm, a.address_norm, b.address_norm,
                                           a.postal_code, b.postal_code)
        grp = "C" if r.g_C else "D" if r.g_D else "A_acc" if r.g_A_acc else "B"
        rows.append((grp, r.s1_id, r.cand_id, r.z, num, nm, " ".join(extra), " ".join(missing)))
    df = pd.DataFrame(rows, columns=["group", "s1_id", "cand_id", "z", "num_pattern", "name_pattern", "extra", "missing"])
    df.to_parquet(os.path.join(REPORT_DIR, "singleton_forensics_diffs.parquet"))

    out = {"note": "A_acc subsampled to 8000, B to 4000 (uniform); C and D complete",
           "num_pattern_pct": {}, "name_pattern_pct": {}, "joint_pct": {}}
    for g, d in df.groupby("group"):
        out["num_pattern_pct"][g] = (100 * d["num_pattern"].value_counts(normalize=True)).round(1).to_dict()
        out["name_pattern_pct"][g] = (100 * d["name_pattern"].value_counts(normalize=True)).round(1).to_dict()
        j = d["num_pattern"].isin(["numbers_differ", "number_added_by_cand", "numbers_cand_subset"])
        out["joint_pct"][g] = {
            "number_changed_or_added": round(100 * float(j.mean()), 1),
            "name_adds_or_substitutes_token": round(100 * float(d["name_pattern"].isin(["cand_adds_token", "token_substituted"]).mean()), 1),
            "both": round(100 * float((j & d["name_pattern"].isin(["cand_adds_token", "token_substituted"])).mean()), 1)}
    # vocabulary of tokens the candidate ADDS to the name (decoy vocabulary check)
    for g in ("A_acc", "C", "D"):
        c = Counter(t for e in df.loc[df["group"] == g, "extra"] for t in e.split())
        n = (df["group"] == g).sum()
        out.setdefault("top_added_name_tokens_per100pairs", {})[g] = {k: round(100 * v / n, 1) for k, v in c.most_common(25)}
    with open(os.path.join(REPORT_DIR, "singleton_forensics_diffs.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))
    categorize()


if __name__ == "__main__":
    main()
