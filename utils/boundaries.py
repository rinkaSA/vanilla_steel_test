import re
import pandas as pd


def _normalize(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("×", "*").replace("x", "*")
    return re.sub(r"\s+", " ", s)

def _has_any_digit(s: str) -> bool:
    return bool(re.search(r"\d", s or ""))

def _find_unit(s: str):
    """
    Extract units even if glued to numbers (27J, 60HRC) or spaced (540-720 MPa).
    Returns (string_without_unit, unit or None).
    """
    for u in ["MPa", "HB", "HV", "HRC", "%", "J"]:
        # glued to a number
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{re.escape(u)}\b", s)
        if m:
            s2 = s[:m.start(1)] + m.group(1) + s[m.end():]
            return s2.strip(), u
        # standalone token
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(u)}(?![A-Za-z0-9])", s)
        if m:
            s2 = (s[:m.start()] + s[m.end():]).strip()
            return s2, u
    return s, None

def _find_tempC(s: str):
    m = re.search(r"\bat\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*C\b", s, re.IGNORECASE)
    if not m: return s, None
    temp = float(m.group(1))
    s2 = (s[:m.start()] + s[m.end():]).strip()
    return s2, temp

def _extract_parenthetical(s: str):
    m = re.search(r"\(([^)]+)\)\s*$", s)
    if not m: return s, None
    txt = m.group(1).strip()
    return s[:m.start()].strip(), txt

def _parse_range(s: str):
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)", s)
    if not m: return None
    a, b = float(m.group(1)), float(m.group(2))
    return (a, b) if a <= b else (b, a)

def _parse_single_number(s: str):
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def _extract_vars(expr: str):
    return re.findall(r"[A-Za-z]\w*", expr or "")

def _eval_formula(expr: str, context: dict | None):
    if not context: return None
    def repl(m):
        var = m.group(0)
        for k, v in context.items():
            if k.lower() == var.lower(): return str(v)
        return var
    expr_norm = re.sub(r"[A-Za-z]\w*", repl, expr)
    try:
        val = eval(expr_norm, {"__builtins__": {}})
        return float(val) if isinstance(val, (int, float)) else None
    except Exception:
        return None

def _upper_from_cell(s: str | None):
    if s is None: return None
    t = _normalize(s)
    r = _parse_range(t)
    if r: return r[1]
    if "<=" in t or ">=" in t:
        return _parse_single_number(t.replace("<=", " ").replace(">=", " "))
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*", t)
    return float(m.group(1)) if m else None

def _symbol_map_from_columns(columns) -> dict:
    mp = {}
    for col in columns:
        m = re.search(r"\(([^)]+)\)", str(col))
        if m:
            sym = m.group(1).strip()
            mp.setdefault(sym, col)
    return mp


def parse_cell(
    text: str,
    *,
    allow_temp: bool = False,
    keep_unit: bool = False,
    context: dict | None = None,
    collect_additional: bool = False,
    allow_formula: bool = False,      # only for Titanium
    allowed_symbols: set[str] | None = None,
    single_to_mid_only: bool = False, # Impact toughness single -> mid only
    round_mid_4: bool = True,         # round mid to 4 decimals
):
    raw = text
    s = _normalize(text)

    # extras first
    additional = None
    if collect_additional:
        s, additional = _extract_parenthetical(s)

    tempC = None
    if allow_temp:
        s, tempC = _find_tempC(s)

    s, unit = _find_unit(s)  # always strip; return only when keep_unit=True

    # non-numeric, non-formula -> NaNs
    if not _has_any_digit(s):
        out = {"raw": raw, "comp": None, "min": None, "max": None, "mid": None, "formula": None}
        if keep_unit: out["unit"] = None
        if allow_temp: out["tempC"] = None
        if collect_additional: out["additional"] = None
        return out

    comp = None
    if "<=" in s:
        comp = "<="; s = s.replace("<=", " ")
    elif ">=" in s:
        comp = ">="; s = s.replace(">=", " ")
    s = s.strip()

    # range
    r = _parse_range(s)
    if r:
        lo, hi = r
        mid = (lo + hi) / 2.0
        if round_mid_4 and mid is not None: mid = round(mid, 4)
        out = {"raw": raw, "comp": comp, "min": lo, "max": hi, "mid": mid}
        if keep_unit: out["unit"] = unit
        if allow_temp and tempC is not None: out["tempC"] = tempC
        if collect_additional and additional is not None: out["additional"] = additional
        return out

    num = _parse_single_number(s)

    # single with comparator
    if num is not None and comp in ("<=", ">="):
        out = {"raw": raw, "comp": comp, "min": None, "max": None, "mid": None}
        if comp == "<=": out["max"] = num
        else:            out["min"] = num
        if keep_unit: out["unit"] = unit
        if allow_temp and tempC is not None: out["tempC"] = tempC
        if collect_additional and additional is not None: out["additional"] = additional
        return out

    # formulas (only if allowed and references known symbols)
    if allow_formula:
        vars_in = _extract_vars(s)
        if any(v in (allowed_symbols or set()) for v in vars_in):
            val = _eval_formula(s, context)
            out = {"raw": raw, "comp": comp, "min": None, "max": None, "mid": None, "formula": s}
            if val is not None:
                if comp == ">=": out["min"] = max(0.0, val)
                else:            out["max"] = max(0.0, val)
            if keep_unit: out["unit"] = unit
            if allow_temp and tempC is not None: out["tempC"] = tempC
            if collect_additional and additional is not None: out["additional"] = additional
            return out

    # exact single number (no comparator)
    if num is not None:
        if single_to_mid_only:
            out = {"raw": raw, "comp": comp, "min": None, "max": None, "mid": num}
        else:
            out = {"raw": raw, "comp": comp, "min": num, "max": num, "mid": num}
        if round_mid_4 and out["mid"] is not None: out["mid"] = round(out["mid"], 4)
        if keep_unit: out["unit"] = unit
        if allow_temp and tempC is not None: out["tempC"] = tempC
        if collect_additional and additional is not None: out["additional"] = additional
        return out

    # fallback
    out = {"raw": raw, "comp": comp, "min": None, "max": None, "mid": None, "formula": None}
    if keep_unit: out["unit"] = unit
    if allow_temp: out["tempC"] = tempC
    if collect_additional and additional is not None: out["additional"] = additional
    return out

def expand_specs(
    df: pd.DataFrame,
    *,
    temp_cols: set[str],
    unit_cols: set[str],
    formula_cols: set[str],           
    single_mid_only_cols: set[str],  
    skip_cols: set[str] = None,      
) -> pd.DataFrame:
    """
    Produces <col>_raw, <col>_comp, <col>_min, <col>_max, <col>_mid, [<col>_unit], [<col>_tempC], [<col>_formula], [<col>_additional]
    - Only `unit_cols` emit _unit; only `temp_cols` emit _tempC; only `formula_cols` emit _formula (Titanium).
    - `single_mid_only_cols`: single numbers (no comparator) go to _mid only (min/max NaN).
    - Columns in `skip_cols` are copied through untouched.
    - _mid is rounded to 4 decimals.
    """
    temp_cols = set(temp_cols or set())
    unit_cols = set(unit_cols or set())
    formula_cols = set(formula_cols or set())
    single_mid_only_cols = set(single_mid_only_cols or set())
    skip_cols = set(skip_cols or set())

    sym_map = _symbol_map_from_columns(df.columns)
    allowed_symbols = set(sym_map.keys())

    parts = []
    for col in df.columns:
        if col in skip_cols:
            parts.append(df[[col]]) 
            continue

        allow_temp = col in temp_cols
        keep_unit = col in unit_cols
        allow_formula = col in formula_cols
        single_to_mid_only = col in single_mid_only_cols
        collect_additional = ("Hardness" in str(col))

        rows = []
        for i, val in df[col].items():
            # per-row context only if formulas allowed
            ctx = None
            if allow_formula:
                vars_in = _extract_vars(_normalize(val))
                if any(v in allowed_symbols for v in vars_in):
                    ctx = {}
                    for v in vars_in:
                        src_col = sym_map.get(v)
                        if not src_col: continue
                        src_raw = df.loc[i, src_col] if src_col in df.columns else None
                        upper = _upper_from_cell(src_raw)
                        if upper is not None: ctx[v] = upper
                    if not ctx: ctx = None

            parsed = parse_cell(
                val,
                allow_temp=allow_temp,
                keep_unit=keep_unit,
                context=ctx,
                collect_additional=collect_additional,
                allow_formula=allow_formula,
                allowed_symbols=allowed_symbols,
                single_to_mid_only=single_to_mid_only,
                round_mid_4=True,
            )
            rows.append(parsed)

        part = pd.DataFrame(rows).rename(columns={
            "raw": f"{col}_raw",
            "comp": f"{col}_comp",
            "min": f"{col}_min",
            "max": f"{col}_max",
            "mid": f"{col}_mid",
            "unit": f"{col}_unit",
            "tempC": f"{col}_tempC",
            "formula": f"{col}_formula",
            "additional": f"{col}_additional",
        })

        # drop columns that aren't enabled for this col
        if not allow_temp and f"{col}_tempC" in part: part.drop(columns=[f"{col}_tempC"], inplace=True)
        if not keep_unit and f"{col}_unit" in part:   part.drop(columns=[f"{col}_unit"], inplace=True)
        if not allow_formula and f"{col}_formula" in part: part.drop(columns=[f"{col}_formula"], inplace=True)
        if not collect_additional and f"{col}_additional" in part: part.drop(columns=[f"{col}_additional"], inplace=True)

        parts.append(part)

    return pd.concat(parts, axis=1)
