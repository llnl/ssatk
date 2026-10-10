import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional

def _cast_value(raw: str, kind: str) -> object:
    """
    Convert a fixed-width slice to the requested type.
    Supported kinds: 'str', 'int', 'float', 'tleexp'
    - 'tleexp' parses TLE exponent format like '16538-3' -> 0.00016538
    """
    s = raw.strip()
    if kind == "str":
        return s
    if s == "" or s == ".":
        return np.nan

    if kind == "int":
        try:
            return int(s)
        except Exception:
            return np.nan

    if kind == "float":
        # Accept leading '.' form (e.g., '.00020137')
        try:
            if s.startswith("."):
                return float("0" + s)
            return float(s)
        except Exception:
            return np.nan

    if kind == "tleexp":
        # TLE exponent fields encode mantissa without a decimal and a 1- or 2-digit exponent:
        # '16538-3' => mantissa 0.16538, exponent -3
        # '00000-0' => 0.0
        t = s.replace(" ", "")
        if t == "" or set(t) <= {"0"} or t.endswith("-0") and set(t[:-2]) <= {"0"}:
            return 0.0
        # split mantissa and exponent by the last sign (+/-) near the end
        # prefer 1-digit exponent; fall back to 2-digit if needed
        if len(t) >= 2 and t[-2] in "+-":
            mant, exps = t[:-2], t[-2:]
        elif len(t) >= 3 and t[-3] in "+-":
            mant, exps = t[:-3], t[-3:]
        else:
            # best effort: assume last char is exponent without sign
            mant, exps = t[:-1], "+" + t[-1]

        m_sign = -1 if mant.startswith("-") else 1
        mant_digits = mant.lstrip("+-")
        # normalize to 5 digits if shorter
        mant_digits = mant_digits.rjust(5, "0")[:5]
        try:
            m = m_sign * float("0." + mant_digits)
            e = int(exps)
            return m * np.power(10.0, e)
        except Exception:
            return np.nan

    raise ValueError(f"Unknown kind: {kind}")


def parse_fixed_width_file(
    file_path: str,
    fields: List[Dict],
    record_lines: int = 1,
    line_selector: Optional[Callable[[str], bool]] = None,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> pd.DataFrame:
    """
    Generic line-by-line, bit-by-bit fixed-width parser.

    Parameters
    ----------
    file_path : path to the text file.
    fields : list of dicts describing slices. Each dict:
        {
          'name': 'column_name',
          'line': 0,                 # 0-based line index within a record
          'start': 0, 'end': 10,     # 0-based [start:end) slice
          'type': 'str'|'int'|'float'|'tleexp'
        }
    record_lines : number of lines that form one logical record (e.g., 2 for TLE).
    line_selector : optional callable(line) -> bool to include only some lines
                    when record_lines == 1. Ignored otherwise.
    encoding, errors : file decoding controls.

    Returns
    -------
    pandas.DataFrame with one row per record and one column per field.
    """
    rows = []
    with open(file_path, "r", encoding=encoding, errors=errors) as f:
        if record_lines == 1:
            for raw in f:
                line = raw.rstrip("\n")
                if line_selector and not line_selector(line):
                    continue
                row = {}
                for spec in fields:
                    s = line[spec["start"]:spec["end"]]
                    row[spec["name"]] = _cast_value(s, spec.get("type", "str"))
                rows.append(row)
        else:
            # group into fixed-size records
            buf = []
            for raw in f:
                line = raw.rstrip("\n")
                if line.strip() == "":
                    continue
                buf.append(line)
                if len(buf) == record_lines:
                    row = {}
                    for spec in fields:
                        li = spec["line"]
                        if li < 0 or li >= record_lines:
                            raise IndexError("field 'line' out of range")
                        src = buf[li]
                        s = src[spec["start"]:spec["end"]]
                        row[spec["name"]] = _cast_value(s, spec.get("type", "str"))
                    rows.append(row)
                    buf = []
            # ignore incomplete trailing record

    return pd.DataFrame(rows)


# --------- Ready-to-use schema for classic 2-line TLE ---------
# Column ranges below are 0-based [start:end), matching the standard TLE spec.
TLE_FIELDS: List[Dict] = [
    # Line 1
    {"name": "satnum",           "line": 0, "start": 2,  "end": 7,  "type": "int"},
    {"name": "classification",   "line": 0, "start": 7,  "end": 8,  "type": "str"},
    {"name": "intldes_year",     "line": 0, "start": 9,  "end": 11, "type": "int"},
    {"name": "intldes_launch",   "line": 0, "start": 11, "end": 14, "type": "int"},
    {"name": "intldes_piece",    "line": 0, "start": 14, "end": 17, "type": "str"},
    {"name": "epoch_year",       "line": 0, "start": 18, "end": 20, "type": "int"},
    {"name": "epoch_day",        "line": 0, "start": 20, "end": 32, "type": "float"},
    {"name": "ndot_over_2",      "line": 0, "start": 33, "end": 43, "type": "float"},
    {"name": "nddot_over_6",     "line": 0, "start": 44, "end": 52, "type": "tleexp"},
    {"name": "bstar",            "line": 0, "start": 53, "end": 61, "type": "tleexp"},
    {"name": "elset_type",       "line": 0, "start": 62, "end": 63, "type": "int"},
    {"name": "elset_num",        "line": 0, "start": 64, "end": 68, "type": "int"},
    # Line 2
    {"name": "inclination_deg",  "line": 1, "start": 8,  "end": 16, "type": "float"},
    {"name": "raan_deg",         "line": 1, "start": 17, "end": 25, "type": "float"},
    {"name": "eccentricity",     "line": 1, "start": 26, "end": 33, "type": "float"},  # implied decimal
    {"name": "arg_perigee_deg",  "line": 1, "start": 34, "end": 42, "type": "float"},
    {"name": "mean_anomaly_deg", "line": 1, "start": 43, "end": 51, "type": "float"},
    {"name": "mean_motion",      "line": 1, "start": 52, "end": 63, "type": "float"},
    {"name": "rev_number",       "line": 1, "start": 63, "end": 68, "type": "int"},
]

def read_3le_by_bit(file_path: str) -> pd.DataFrame:
    """
    Robust TLE reader that ignores optional name (line '0') rows and
    pairs each line '1' with the following line '2' having the same satnum.
    Then applies fixed-width slices defined in TLE_FIELDS.

    Returns
    -------
    pandas.DataFrame
    """
    def _get_satnum(line: str) -> str:
        # columns 3-7 (1-based) => [2:7] zero-based
        return line[2:7].strip()

    def _safe_slice(text: str, start: int, end: int) -> str:
        if start >= len(text):
            return ""
        return text[start:min(end, len(text))]

    # Collect well-formed (L1, L2) pairs
    pairs = []
    pending_l1 = None
    pending_sat = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line:
                continue

            # Strip UTF-8 BOM if present on first line
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")

            tag = line[:1]
            if tag == "0":
                # optional name line; skip for pairing (you can capture it separately if needed)
                continue

            if tag == "1":
                # start a new record
                pending_l1 = line
                pending_sat = _get_satnum(line)
                continue

            if tag == "2" and pending_l1 is not None:
                sat2 = _get_satnum(line)
                if sat2 == pending_sat:
                    pairs.append((pending_l1, line))
                # whether matched or not, clear pending to avoid cross-pairing
                pending_l1 = None
                pending_sat = None

            # Any other tags are ignored

    # Now parse each pair using the fixed-width schema
    rows = []
    for l1, l2 in pairs:
        row = {}
        for spec in TLE_FIELDS:
            src = l1 if spec["line"] == 0 else l2
            s = _safe_slice(src, spec["start"], spec["end"])
            row[spec["name"]] = _cast_value(s, spec.get("type", "str"))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Eccentricity is given without a leading decimal in TLE line 2.
    if "eccentricity" in df.columns:
        with np.errstate(all="ignore"):
            df["eccentricity"] = np.where(
                df["eccentricity"].notna(),
                df["eccentricity"] * 1e-7,  # 7 digits in classic TLE
                np.nan,
            )

    return df
