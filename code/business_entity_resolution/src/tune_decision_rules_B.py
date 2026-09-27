"""
TUNE-only decision-rule analysis for the frozen number-aware model B.

The frozen B result on the 10K VAL sample is NOT touched: this script reads only
training-side rows (FIT/TUNE) and asserts that no VAL S1 is loaded.

Two pre-stated, single-parameter hypotheses (1-D search over the same T_GRID that
selected B's global threshold; nothing else is searched):
  H1 multi-match recovery: S1 whose top candidate clears the frozen t (=8.3) is a
     confirmed match; its further candidates are accepted at t2 <= t.
     Singleton handling is unchanged (a singleton still needs z1 >= t).
  H2 zero-prediction recovery: S1 with no candidate >= t gets ONLY its top
     candidate if z1 >= t_low AND that pair has house-number agreement
     (num_hn_head_equal = 1) and no number contradiction.
Both are reported on TUNE (selection split) with a paired bootstrap vs frozen B,
and on FIT as a secondary look (FIT is in-sample for the LR, so optimistic).
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tune_decision_rules import T_GRID, Part, f05_vec, paired_bootstrap

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..", "..")
FEAT_DIR = os.path.join(ROOT, "data_cache", "dev_features")
SCORE_DIR = os.path.join(ROOT, "data_cache", "dev_scores")
REPORT_DIR = os.path.join(ROOT, "reports")
PART_COLS = ["name_nosuffix_exact", "address_missing_cand", "n_channels_hit", "postal_equal", "address_missing_s1"]
T_FROZEN = 8.3


def load_training_side():
    s1 = pd.read_parquet(os.path.join(SCORE_DIR, "D3_s1_table.parquet"))
    val_ids = set(s1.loc[s1["part"] == "val", "entity_id"])
    base = pq.read_table(os.path.join(FEAT_DIR, "D3_train_features.parquet"),
                         columns=["s1_id", "cand_id", "label"] + PART_COLS).to_pandas()
    num = pd.read_parquet(os.path.join(FEAT_DIR, "D3_train_numfeat.parquet"),
                          columns=["num_hn_head_equal", "num_hn_head_mismatch", "num_compound_mismatch",
                                   "num_token_contradiction"])
    z = pd.read_parquet(os.path.join(SCORE_DIR, "D3B_num_train_z.parquet"))["z_train"].values
    assert len(base) == len(num) == len(z)
    base["s1_id"] = base["s1_id"].astype(str)
    base["cand_id"] = base["cand_id"].astype(str)
    assert not set(base["s1_id"].unique()) & val_ids, "VAL S1 present in training-side rows"
    base["z"] = z
    base["num_ok"] = ((num["num_hn_head_equal"] == 1) & (num["num_hn_head_mismatch"] == 0)
                      & (num["num_compound_mismatch"] == 0) & (num["num_token_contradiction"] == 0)).values
    return s1, base


class RulePart:
    def __init__(self, s1_df, rows):
        self.p = Part(s1_df, rows[["s1_id", "cand_id", "label", "z"] + PART_COLS])
        # num_ok aligned with Part's internal (s1, -z) sort, restricted to z >= Z_FLOOR rows
        idx = pd.Series(np.arange(self.p.n), index=self.p.s1_ids)
        s = idx.reindex(rows["s1_id"].values).values.astype(np.int64)
        order = np.lexsort((-rows["z"].values, s))
        z_sorted = rows["z"].values[order]
        keep = z_sorted >= 2.0
        self.r_ok = rows["num_ok"].values[order][keep]
        assert len(self.r_ok) == len(self.p.r_z)
        first = np.r_[True, self.p.r_s[1:] != self.p.r_s[:-1]]
        self.is_top = first                                      # rows sorted by s1 then -z

    def accepted(self, rule, param=None):
        """Row-level acceptance mask over self.p's pruned, (s1, -z)-sorted rows."""
        p = self.p
        z1 = p.z1
        if rule == "B":
            acc = p.r_z >= T_FROZEN
        elif rule == "H1":
            confirmed = z1 >= T_FROZEN
            acc = np.where(confirmed[p.r_s], p.r_z >= param, p.r_z >= T_FROZEN)
        elif rule == "H2":
            empty = z1 < T_FROZEN
            acc = (p.r_z >= T_FROZEN) | (empty[p.r_s] & self.is_top & (p.r_z >= param) & self.r_ok)
        return acc

    def evaluate(self, rule, param=None):
        p = self.p
        acc = self.accepted(rule, param)
        n_pred = np.bincount(p.r_s[acc], minlength=p.n)
        tp = np.bincount(p.r_s[acc], weights=p.r_label[acc], minlength=p.n)
        P, R, F = f05_vec(p.n_true, n_pred, tp)
        single = p.n_true == 0
        matched = ~single
        multi = p.n_true >= 2
        return {"macro_f05": round(float(F.mean()), 4), "macro_precision": round(float(P.mean()), 4),
                "macro_recall": round(float(R.mean()), 4),
                "singleton_fp_rate_pct": round(100 * float((n_pred[single] > 0).mean()), 2),
                "matched_zero_pred_pct": round(100 * float((n_pred[matched] == 0).mean()), 2),
                "multi_match_recall": round(float(R[multi].mean()), 4),
                "avg_pred_per_s1": round(float(n_pred.mean()), 3)}, F


def main():
    s1, rows = load_training_side()
    parts = {}
    for name in ("tune", "fit"):
        ids = s1[s1["part"] == name].reset_index(drop=True)
        parts[name] = RulePart(ids, rows[rows["s1_id"].isin(set(ids["entity_id"]))].reset_index(drop=True))
    tune = parts["tune"]
    out = {"note": "TUNE = selection split; FIT is in-sample for the LR (optimistic). VAL not loaded.",
           "frozen_B_threshold": T_FROZEN, "hypotheses": {}}
    base_t, F_base = tune.evaluate("B")
    base_f, F_base_fit = parts["fit"].evaluate("B")
    out["frozen_B"] = {"tune": base_t, "fit": base_f}
    for rule, grid in (("H1", [t for t in T_GRID if t <= T_FROZEN]), ("H2", [t for t in T_GRID if t < T_FROZEN])):
        curve = {float(t): tune.evaluate(rule, t)[0]["macro_f05"] for t in grid}
        best = max(curve, key=curve.get)
        res_t, F_t = tune.evaluate(rule, best)
        res_f, F_f = parts["fit"].evaluate(rule, best)
        out["hypotheses"][rule] = {
            "selected_param_on_tune": best,
            "tune": res_t, "tune_delta_vs_B": paired_bootstrap(F_base, F_t),
            "fit_secondary": res_f, "fit_delta_vs_B": paired_bootstrap(F_base_fit, F_f),
            "tune_curve_f05": {k: v for k, v in curve.items() if abs(k - best) <= 1.0 or k in (min(curve), T_FROZEN)},
        }
    with open(os.path.join(REPORT_DIR, "decision_rules_B_tune.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
