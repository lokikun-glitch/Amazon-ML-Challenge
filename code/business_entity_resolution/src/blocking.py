"""
Phase 4: Blocking / candidate generation framework.

Memory-safe inverted-index retrieval, refactored per review feedback:

  - min_df=1 is the default and is NOT treated as noise-removal: a hapax
    business-specific token ("Zypherion", "Qorvatek") can be an excellent
    blocking key precisely because it is rare. The frequency control that
    matters is max_df, which bounds postings-list size for very common
    tokens ("limited", "road") -- a standalone-blocking weakness, not a
    reason to delete them from any representation.
  - df-counting (one pass, tokenize once) is decoupled from postings-building
    (one pass per max_df threshold) so a max_df sweep does not repeat the
    expensive full tokenize+count pass for every threshold -- only the much
    cheaper "which keys survive the cutoff + append row position" pass.
  - Query-time supports both OR (union of any selected token's postings) and
    minimum-overlap (candidate must hit >= N of the selected query tokens),
    computed from the already-retrieved union via a Counter, not by any
    global pairwise comparison.
  - Country is folded into every index key as `(row.country, token)` -- a
    hard partition validated by 100% country agreement across all 7.6M
    training ground-truth edges, but generic: whatever string `country`
    holds becomes a partition key, so France/other future labels need no
    special-casing.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except ImportError:
    _PROC = None


def rss_mb():
    """Actual process resident-set size in MB (not a Python-object-only estimate)."""
    if _PROC is None:
        return None
    return _PROC.memory_info().rss / 1e6


class ExactIndex:
    """country + exact string key -> candidate row positions (pandas groupby, no explode)."""

    def __init__(self, name):
        self.name = name
        self.postings = {}

    def fit(self, country: pd.Series, key: pd.Series, max_df=None):
        """
        max_df: drop (country, key) groups whose postings list exceeds this size.
        For an exact-match index, group size *is* the document frequency directly
        (no separate counting pass needed, unlike TokenIndex) -- e.g. a common
        street number like "1" or "100" shared by tens of thousands of unrelated
        records is exactly the case this caps.
        """
        t0 = time.time()
        rss0 = rss_mb()
        df = pd.DataFrame({"country": country.values, "key": key.values})
        df = df[df["key"] != ""]
        grouped = df.groupby(["country", "key"], sort=False).indices
        orig_pos = df.index.to_numpy()
        if max_df is None:
            self.postings = {k: orig_pos[v].astype(np.int32) for k, v in grouped.items()}
        else:
            self.postings = {k: orig_pos[v].astype(np.int32) for k, v in grouped.items() if len(v) <= max_df}
        self.max_df = max_df
        self.n_distinct_keys_seen = len(grouped)
        self.n_distinct_keys_kept = len(self.postings)
        self.fit_time = time.time() - t0
        rss1 = rss_mb()
        self.fit_rss_delta_mb = (rss1 - rss0) if (rss0 is not None) else None
        return self

    def query(self, country, key):
        if not key:
            return np.empty(0, dtype=np.int32)
        return self.postings.get((country, key), np.empty(0, dtype=np.int32))

    def postings_array_mb(self):
        return sum(arr.nbytes for arr in self.postings.values()) / 1e6


class TokenIndex:
    """
    country + token -> candidate row positions, frequency-pruned by max_df.

    fit_df_counts() and build_postings() are separate so a max_df sweep does
    the expensive tokenize+count pass exactly once and repeats only the
    cheap postings-assembly pass per threshold.
    """

    def __init__(self, name):
        self.name = name
        self.df_counts = {}
        self.postings = {}
        self.max_df = None
        self.min_df = None

    def fit_df_counts(self, country: pd.Series, token_strings: pd.Series):
        t0 = time.time()
        self._countries = country.values  # kept resident for build_postings() reuse
        self._token_strings = token_strings.values
        df_counter = Counter()
        for c, s in zip(self._countries, self._token_strings):
            if not s:
                continue
            for t in set(s.split()):
                df_counter[(c, t)] += 1
        self.df_counts = df_counter
        self.df_count_time = time.time() - t0
        return self

    def build_postings(self, min_df=1, max_df=3000):
        """Rebuild postings for a new (min_df, max_df) threshold; O(rows), no re-tokenize-count."""
        t0 = time.time()
        keep = {k for k, v in self.df_counts.items() if min_df <= v <= max_df}
        tmp = {}
        for i in range(len(self._countries)):
            s = self._token_strings[i]
            if not s:
                continue
            c = self._countries[i]
            for t in set(s.split()):
                key = (c, t)
                if key in keep:
                    tmp.setdefault(key, []).append(i)
        self.postings = {k: np.asarray(v, dtype=np.int32) for k, v in tmp.items()}
        self.min_df, self.max_df = min_df, max_df
        self.n_distinct_keys_seen = len(self.df_counts)
        self.n_distinct_keys_kept = len(keep)
        self.build_postings_time = time.time() - t0
        return self

    def fit(self, country: pd.Series, token_strings: pd.Series, min_df=1, max_df=3000):
        self.fit_df_counts(country, token_strings)
        self.build_postings(min_df=min_df, max_df=max_df)
        self.fit_time = self.df_count_time + self.build_postings_time
        return self

    def release_raw(self):
        """Drop the resident raw column once no more max_df rebuilds are needed."""
        self._countries = None
        self._token_strings = None

    def query(self, country, token_string, max_query_tokens=6, min_overlap=1):
        """
        OR retrieval when min_overlap=1 (default): candidate if it shares >=1
        selected token. min_overlap>1: candidate must hit >= N of the
        selected top-K (rarest-first) query tokens -- computed from the
        already-retrieved union via a Counter, no pairwise comparison.
        """
        if not token_string:
            return np.empty(0, dtype=np.int32)
        toks = list(set(token_string.split()))
        toks.sort(key=lambda t: self.df_counts.get((country, t), 10 ** 9))
        toks = toks[:max_query_tokens]
        if not toks:
            return np.empty(0, dtype=np.int32)
        if min_overlap <= 1:
            out = [self.postings[(country, t)] for t in toks if (country, t) in self.postings]
            if not out:
                return np.empty(0, dtype=np.int32)
            return np.unique(np.concatenate(out))
        counts = Counter()
        for t in toks:
            arr = self.postings.get((country, t))
            if arr is not None:
                counts.update(arr.tolist())
        if not counts:
            return np.empty(0, dtype=np.int32)
        return np.array([pos for pos, c in counts.items() if c >= min_overlap], dtype=np.int32)

    def postings_array_mb(self):
        return sum(arr.nbytes for arr in self.postings.values()) / 1e6


def char_ngram_string(s: str, n: int = 3) -> str:
    """Space-joined n-grams so TokenIndex can reuse its split()-based machinery."""
    s = s.replace(" ", "")
    if len(s) < n:
        return s
    return " ".join(s[i:i + n] for i in range(len(s) - n + 1))


def build_ngram_series(strings: pd.Series, n: int = 3) -> pd.Series:
    return strings.map(lambda s: char_ngram_string(s, n) if s else "")
