"""
Deterministic address-number extraction (house / compound / unit / postal).

Separate from normalization.py on purpose: the existing `postal_code` and
`street_number` columns feed the frozen baseline features and are left as-is.
Audit of those two fields (see reports/number_features_experiment.md):
  * postal_code = last 5-6 digit run anywhere in the raw string. The corpus has
    NO US ZIP codes and no standalone Indian PINs, so for US it is almost always
    a 5-digit house/unit/road number (98% equal to street_number when present)
    and for zero-padded numbers ("005424 ...") a 6-digit house number.
  * street_number = first all-digit token of address_norm, where punctuation has
    already become whitespace: "26/244" -> "26", "6-3-668/10/4" -> "6",
    "B-1/307" -> "1"; leading zeros kept ("002798" != "2798"); letter-suffixed
    numbers skipped ("5527C" -> next digit token); ordinals/sector/floor numbers
    can be picked up ("Level 5" -> "5").

What this module extracts (uppercase raw address, split on commas):
  number expression  maximal run like 26/244, 6-3-668/10/4, B-1/307, 85P3, 5527C,
                     FK-11&12, 115-48, 1026 1/2. Ordinals (3RD, 14TH) are not numbers.
  label              the keyword immediately before an expression:
                     HOUSE  (H.No, D.No, Door No, House No, No, Plot No, Pno, Prop No, #)
                     UNIT   (Flat, Fno, Shop, Office, Room, Rno, Unit, Apt, Suite, Ste)
                     PARCEL (S.No, Sno, Sr No, Survey No, SF No, Khasra, Kh No, Khata, Gat, CTS)
                     CONTEXT(Sector, Phase, Ward, Floor, Level, Block, Bldg, Road/Street/Gali No,
                             Highway, NH, Route, County Road, FM, PMB, PO Box, ...)
  primary_house_number  first HOUSE-labelled expression; else first unlabelled
                        expression at the start of a segment; else first PARCEL one.
                        Canonical form: separators kept, spaces dropped, leading
                        zeros stripped per atom ("##0018" -> "18", "00229/A" -> "229/A").
  house_head / house_atoms  first numeric atom / all numeric atoms of the primary.
  compound_number    the primary when it has >= 2 numeric atoms (e.g. "26/244"), else "".
  unit_number        first UNIT-labelled value (must contain a digit or be one letter);
                     a segment that is only "# N" (after a street segment) is a unit.
  postal_code        US: only a 5-digit code right after a state abbreviation/name at
                     the END of the string (never observed in this corpus -> ""), so a
                     ZIP can never be read as a house number and vice versa.
                     India: 6-digit PIN, or "624 103" split form after '-' / 'PIN'.
  number_tokens      all numeric atoms of all expressions (postal excluded), leading
                     zeros stripped.
Ambiguous material (C/O, W/O, unlabelled mid-segment numbers, street-name numbers)
is never promoted to house/unit; it only contributes to number_tokens.
"""
import re
import unicodedata

_NULLS = re.compile(r"<NULL>|\bNULL\b|\bN/A\b")

# expression: optional 1-2 letter prefix (B-1, SB-706, C101), atom, then joined parts
_ATOM = r"\d+(?:[A-Z]\d+)?[A-Z]?"
_PART = r"(?:\s?[/\-&]\s?(?:[A-Z]{1,2}\s?-?\s?)?" + _ATOM + r"|\s?[/\-]\s?[A-Z](?![A-Z0-9]))"
_EXPR = re.compile(r"(?<![A-Z0-9])((?:[A-Z]{1,2}\s?-\s?|[A-Z]{1,2}(?=\d))?" + _ATOM + r"(?:" + _PART + r")*)(?![A-Z0-9])")
_FRACTION = re.compile(r"^\s+(1/[234])(?![0-9])")          # "1026 1/2" -> one US house number
_DIGITS = re.compile(r"\d+")


def _lab(*words):
    # words may contain '.' and spaces between letters; allow optional dots/spaces
    alts = []
    for w in words:
        parts = [re.escape(p) for p in w.split()]
        alts.append(r"\.?\s*".join(parts))
    return r"(?:" + "|".join(sorted(alts, key=len, reverse=True)) + r")"


_LABELS = [
    ("UNIT", _lab("FLAT NO", "FLAT", "FNO", "F NO", "SHOP NO", "SHOP", "OFFICE NO", "OFFICE", "ROOM NO", "ROOM",
                  "R NO", "RNO", "UNIT NO", "UNIT", "APARTMENT", "APT", "SUITE", "STE")),
    ("PARCEL", _lab("SURVEY NO", "SR NO", "S R NO", "S NO", "SNO", "SF NO", "KHASRA NO", "KHASRA", "KH NO",
                    "KHATA NO", "KHATA", "GAT NO", "CTS NO", "SN")),
    ("CONTEXT", _lab("SECTOR", "SEC", "PHASE", "WARD NO", "WARD", "FLOOR", "FLR", "FL", "LEVEL", "BLOCK", "BLDG NO",
                     "BUILDING NO", "BLDG", "BUILDING", "TOWER", "WING", "STAGE", "ROAD NO", "RD NO", "STREET NO",
                     "ST NO", "GALI NO", "GALI", "LANE NO", "NATIONAL HIGHWAY NO", "NATIONAL HIGHWAY", "HIGHWAY",
                     "HWY", "NH", "ROUTE", "RTE", "COUNTY ROAD", "CR", "FARM TO MARKET", "FM", "PMB", "PO BOX",
                     "P O BOX", "BOX", "NEW NO", "OLD NO", "MAIN", "CROSS", "PIN", "PINCODE")),
    ("HOUSE", _lab("HOUSE NO", "HOUSENO", "H NO", "HNO", "HN", "D NO", "DNO", "DOOR NO", "DOOR", "PLOT NO",
                   "PLOT", "P NO", "PNO", "PROP NO", "BUNGALOW NO", "BUNGLOW NO", "NO")),
]
_LABEL_RE = re.compile(r"(?<![A-Z0-9])(?:" + "|".join(f"(?P<{k}>{p})" for k, p in _LABELS)
                       + r")(?![A-Z])")
_GAP = re.compile(r"^[\s.:\-#]*$")                       # what may sit between a label and its number

_US_STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
              "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()
_US_STATE_NAMES = ("ALABAMA ALASKA ARIZONA ARKANSAS CALIFORNIA COLORADO CONNECTICUT DELAWARE FLORIDA GEORGIA HAWAII "
                   "IDAHO ILLINOIS INDIANA IOWA KANSAS KENTUCKY LOUISIANA MAINE MARYLAND MASSACHUSETTS MICHIGAN "
                   "MINNESOTA MISSISSIPPI MISSOURI MONTANA NEBRASKA NEVADA OHIO OKLAHOMA OREGON PENNSYLVANIA "
                   "TENNESSEE TEXAS UTAH VERMONT VIRGINIA WASHINGTON WISCONSIN WYOMING").split() + [
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "RHODE ISLAND",
    "SOUTH CAROLINA", "SOUTH DAKOTA", "WEST VIRGINIA"]
# "ST 12345" / "State 12345[-6789]" at the very end of the string. "FL" doubles as the
# floor label ("Fl 13887"), so FL additionally needs a Florida ZIP prefix (32-34).
_US_ZIP = re.compile(r"(?:^|[\s,])(?:(?:" + "|".join(x for x in _US_STATES if x != "FL") + "|"
                     + "|".join(_US_STATE_NAMES) + r")\s*,?\s+(\d{5})|FL\s+(3[2-4]\d{3}))(?:-\d{4})?\s*$")
_IN_PIN = re.compile(r"(?:(?:PIN(?:CODE)?\s*[:\-]?\s*)|[\-–]\s*)([1-9]\d{2})\s?(\d{3})(?![0-9])|(?<![0-9])([1-9]\d{5})(?![0-9])")


def _canon(expr):
    """Canonical number expression: no spaces, leading zeros stripped per atom."""
    e = re.sub(r"\s+", "", expr)
    return _DIGITS.sub(lambda m: m.group(0).lstrip("0") or "0", e)


def _atoms(expr):
    return [a.lstrip("0") or "0" for a in _DIGITS.findall(expr)]


def _prep(addr):
    s = unicodedata.normalize("NFKC", addr or "").upper()
    return _NULLS.sub(" ", s)


def extract_postal(s, country):
    """Returns (postal_code, span) on the prepared (uppercase) string."""
    if country == "US":
        m = _US_ZIP.search(s)
        if not m:
            return "", None
        g = 1 if m.group(1) else 2
        return m.group(g), m.span(g)
    if country == "India":
        found = None
        for m in _IN_PIN.finditer(s):
            found = m                                          # last one wins
        if found:
            code = found.group(3) or (found.group(1) + found.group(2))
            span = found.span(3) if found.group(3) else (found.start(1), found.end(2))
            return code, span
    return "", None


def extract(addr, country):
    s = _prep(addr)
    out = {"primary_house_number": "", "house_head": "", "house_atoms": "", "compound_number": "",
           "unit_number": "", "postal_code": "", "number_tokens": ""}
    if not s.strip():
        return out
    postal, span = extract_postal(s, country)
    out["postal_code"] = postal
    if span:
        s = s[:span[0]] + " " * (span[1] - span[0]) + s[span[1]:]   # mask, keep offsets

    house = lead = parcel = hash_house = None
    unit = None
    tokens = set()
    seen_street_segment = False
    for seg in s.split(","):
        labels = [(m.start(), m.end(), m.lastgroup) for m in _LABEL_RE.finditer(seg)]
        seg_stripped = seg.strip()
        only_hash = re.fullmatch(r"#+\s*([A-Z]?\d+[A-Z]?)", seg_stripped)
        if only_hash and seen_street_segment and unit is None:
            unit = _canon(only_hash.group(1))
        for m in _EXPR.finditer(seg):
            expr, a, b = m.group(1), m.start(1), m.end(1)
            fr = _FRACTION.match(seg[b:])
            c = _canon(expr)
            if fr:                                           # "1026 1/2": fraction belongs to the house number
                c = c + " " + fr.group(1)
                expr = expr + " " + fr.group(1)
            atoms = _atoms(expr)
            tokens.update(atoms)
            before = seg[:a]
            # label = last label whose end is separated from the expression by a gap only
            lab = None
            for ls, le, lk in labels:
                if le <= a and _GAP.match(seg[le:a]):
                    lab = lk
            hashed = "#" in before[-3:] and _GAP.match(before.strip()[-3:] or "")
            at_start = _GAP.match(before) is not None
            if lab == "HOUSE" and house is None:
                house = (c, atoms)
            elif lab == "UNIT":
                if unit is None:
                    unit = c
            elif lab == "PARCEL":
                if parcel is None:
                    parcel = (c, atoms)
            elif lab == "CONTEXT":
                pass
            elif at_start and not only_hash:
                if hashed and hash_house is None:
                    hash_house = (c, atoms)
                elif lead is None:
                    lead = (c, atoms)
        # unit values that are a single letter ("Unit APARTMENT G")
        if unit is None:
            um = re.search(r"(?<![A-Z])(?:UNIT|APT|APARTMENT|SUITE|STE)(?:\s+(?:UNIT|APT|APARTMENT|SUITE|STE))*\s+([A-Z])\s*$", seg)
            if um:
                unit = um.group(1)
        if re.search(r"\d", seg) and re.search(r"[A-Z]{3,}", seg):
            seen_street_segment = True
    prim = house or hash_house or lead or parcel
    if prim:
        out["primary_house_number"] = prim[0]
        out["house_head"] = prim[1][0]
        out["house_atoms"] = "|".join(prim[1])
        if len(prim[1]) >= 2:
            out["compound_number"] = prim[0]
    out["unit_number"] = unit or ""
    out["number_tokens"] = " ".join(sorted(tokens, key=lambda t: (len(t), t)))
    return out


FIELDS = ["primary_house_number", "house_head", "house_atoms", "compound_number",
          "unit_number", "postal_code", "number_tokens"]
