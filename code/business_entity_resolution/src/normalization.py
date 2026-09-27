"""
Phase 3: Normalization module.

Produces multiple, non-destructive representations of business_name and
business_address, reusable by both the blocking stage and the feature stage.

Design principles (per spec):
  - Never collapse to a single aggressively-normalized string; keep several
    representations and let blocking/matching decide which is useful.
  - Legal-suffix removal is a *parallel* representation, not a replacement.
  - Transliteration (unidecode) is a *parallel* representation for cross-script
    matching (observed Devanagari/Kannada vs Latin in the forensic sample),
    never a replacement of the original.
  - Vectorized pandas .str operations are used wherever possible; unidecode
    and NFKC normalization have no vectorized pandas equivalent and fall back
    to a per-element map (still fast: table lookups, no network/IO).
"""
import re
import unicodedata

import pandas as pd
from unidecode import unidecode as _unidecode

# --- legal suffix tokens (generic, not country-conditioned; language-agnostic
# list covering US/India/France style company suffixes seen in the corpus) ---
LEGAL_SUFFIX_TOKENS = frozenset({
    "ltd", "limited", "pvt", "private", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "llp", "lp", "pc", "pllc",
    "gmbh", "sarl", "sas", "sa", "eurl", "srl", "bv", "nv", "ag", "pty",
    "limitada", "spa", "kg", "kft", "oy", "ab", "as", "aps", "sasu", "eirl",
})

# Python's \w does NOT include Unicode combining marks (category M*), so a naive
# [^\w\s] strip mangles Devanagari/other Indic scripts by deleting matras
# (dependent vowel signs) and leaving isolated consonants. Precompute the set of
# combining-mark codepoints in the BMP once and keep them alongside \w and \s.
_MARK_CHARS = "".join(chr(cp) for cp in range(0x10000) if unicodedata.category(chr(cp))[0] == "M")
PUNCT_RE = re.compile(r"[^\w\s" + re.escape(_MARK_CHARS) + r"]", re.UNICODE)
WS_RE = re.compile(r"\s+")
AMP_RE = re.compile(r"&")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")
DIGIT_TOKEN_RE = re.compile(r"^\d+$")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def basic_clean_series(s: pd.Series) -> pd.Series:
    """Vectorized: lowercase, & -> and, strip punctuation, collapse whitespace."""
    s = s.fillna("")
    s = s.map(nfkc)  # no vectorized NFKC in pandas; cheap per-element table op
    s = s.str.lower()
    s = s.str.replace(AMP_RE, " and ", regex=True)
    s = s.str.replace(PUNCT_RE, " ", regex=True)
    s = s.str.replace(WS_RE, " ", regex=True).str.strip()
    return s


def transliterate_series(s: pd.Series) -> pd.Series:
    """Per-element unidecode fallback (no vectorized equivalent). Cheap table lookups."""
    return s.map(lambda x: _unidecode(x) if x else "")


def strip_legal_suffix_tokens(name_norm: str) -> str:
    if not name_norm:
        return name_norm
    toks = [t for t in name_norm.split() if t not in LEGAL_SUFFIX_TOKENS]
    return " ".join(toks)


def strip_legal_suffix_series(s: pd.Series) -> pd.Series:
    return s.map(strip_legal_suffix_tokens)


def extract_postal_tokens(addr_raw: str):
    if not isinstance(addr_raw, str) or not addr_raw:
        return ""
    m = POSTAL_RE.findall(addr_raw)
    return m[-1] if m else ""  # last 5-6 digit run is usually the PIN/ZIP, not a street number


def extract_postal_series(s: pd.Series) -> pd.Series:
    return s.fillna("").map(extract_postal_tokens)


def extract_leading_street_number(addr_norm: str):
    """First all-digit token anywhere in the normalized address (street number heuristic)."""
    if not addr_norm:
        return ""
    for tok in addr_norm.split():
        if DIGIT_TOKEN_RE.match(tok):
            return tok
    return ""


def extract_street_number_series(s: pd.Series) -> pd.Series:
    return s.map(extract_leading_street_number)


def build_normalized_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with columns entity_id, business_name, business_address, country.
    Output: compact DataFrame with entity_id, country, and normalized representations.
    Tokens/char-n-grams are NOT materialized here (computed on the fly by blocking
    to avoid storing large redundant structures) -- only strings are stored.
    """
    out = pd.DataFrame(index=df.index)
    out["entity_id"] = df["entity_id"].values
    out["country"] = df["country"].values  # plain string: keeps parquet schema stable across chunks

    name_norm = basic_clean_series(df["business_name"])
    out["name_norm"] = name_norm
    out["name_no_suffix"] = strip_legal_suffix_series(name_norm)

    name_translit = transliterate_series(df["business_name"].fillna(""))
    name_translit_norm = basic_clean_series(name_translit)
    out["name_translit"] = name_translit_norm

    addr_raw = df["business_address"].fillna("")
    addr_norm = basic_clean_series(addr_raw)
    out["address_norm"] = addr_norm

    addr_translit = transliterate_series(addr_raw)
    out["address_translit"] = basic_clean_series(addr_translit)

    out["postal_code"] = extract_postal_series(addr_raw)
    out["street_number"] = extract_street_number_series(addr_norm)

    return out


def tokset(s: str):
    return set(s.split()) if s else set()


def char_ngrams(s: str, n: int = 3):
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


if __name__ == "__main__":
    # smoke test
    demo = pd.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S3-1"],
        "business_name": ["ABC Pvt. Ltd.", "Ch0ice Next Inc", "राम मार्केटिंग प्राइवेट लिमिटेड"],
        "business_address": ["123 Main St, Delhi, 110001", "005424 OLDE VINTAGE DR, HILLIARD, OH", None],
        "country": ["India", "US", "India"],
    })
    res = build_normalized_frame(demo)
    print(res.to_string())
