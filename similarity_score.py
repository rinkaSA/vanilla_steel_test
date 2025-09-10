import numpy as np
import pandas as pd
import re


DIM_BASES = [
    "thickness", "width", "length", "height",
    "inner_diameter", "outer_diameter", "weight",
]

CAT_FIELDS = ["coating", "finish", "surface_type", "surface_protection", "form"]

RM_COL = "Tensile strength (Rm)_mid"
CP_TOLERANCE = 0.10  

W_DIM, W_CAT, W_CP = 0.60, 0.30, 0.10

GRADE_MULT_EXACT = 1.00
GRADE_MULT_SAME_BASE = 0.95
GRADE_MULT_DIFF_BASE = 0.85

CATEGORY_MATCH_MULT = 1.00
CATEGORY_FALLBACK_MULT = 0.50


def _as_num_pair(iv):
    """Coerce '(2.0, 2.0)' / '[2,2]' / (2,2) / [2,2] → (float, float). Return None if not parseable."""
    if iv is None: 
        return None
    if isinstance(iv, (tuple, list)) and len(iv) == 2:
        return float(iv[0]), float(iv[1])
    if isinstance(iv, str):
        try:
            a, b = iv.strip().strip('()[]').split(',', 1)
            return float(a), float(b)
        except Exception:
            return None
    if isinstance(iv, (int, float))  or isinstance(iv, (float, float)) or isinstance(iv, (float, int)):
        v = float(iv)
        return (v, v)
    return None

def calculate_iou(interval1, interval2):
    """IoU for numeric ranges; handles zero-width as tiny segments; returns 0..1."""
    if interval1 is None or interval2 is None:
        return 0.0
    min1, max1 = interval1
    min2, max2 = interval2
    if min1 == max1:
        min1 -= 1e-6; max1 += 1e-6
    if min2 == max2:
        min2 -= 1e-6; max2 += 1e-6
    inter_min = max(min1, min2)
    inter_max = min(max1, max2)
    if inter_max <= inter_min:
        return 0.0
    inter_len = inter_max - inter_min
    union_len = (max1 - min1) + (max2 - min2) - inter_len
    return inter_len / union_len if union_len > 0 else 0.0


def extract_std_base(s):
    """Return 'EN 10346' from 'EN 10346:2015' (or None). Used only for tie-breaks."""
    if pd.isna(s):
        return None
    m = re.search(r'(EN\s*\d{5})', str(s).upper())
    return m.group(1).replace('  ', ' ') if m else None


def get_interval_from_row(row, base):
    """
    Build an interval for a dimension base:
      1) use '{base}_interval' if present and not NaN (expects (min,max) tuple)
      2) else use '{base}_min'/'{base}_max'; if only one exists, treat as (v, v)
      3) special-case: if base=='length' and only 'length_min' exists, use (v, v)
    Returns None if no usable info.
    """
    #  direct interval column
    col_int = f"{base}_interval"
    if col_int in row and pd.notna(row[col_int]):
        return row[col_int]

    # min/max pair
    col_min, col_max = f"{base}_min", f"{base}_max"
    vmin = row[col_min] if col_min in row else np.nan
    vmax = row[col_max] if col_max in row else np.nan
    if pd.notna(vmin) and pd.notna(vmax):
        return (float(vmin), float(vmax))
    if pd.notna(vmin) and pd.isna(vmax):
        return (float(vmin), float(vmin))
    if pd.isna(vmin) and pd.notna(vmax):
        return (float(vmax), float(vmax))

    #  length_min only
    if base == "length" and "length_min" in row and pd.notna(row["length_min"]):
        v = float(row["length_min"])
        return (v, v)

    return None


def dim_score(q_row, c_row):
    """Mean IoU across all usable dimensions (equal weight)."""
    used_dims, ious = [], []
    for base in DIM_BASES:
        iq = _as_num_pair(get_interval_from_row(q_row, base))
        ic = _as_num_pair(get_interval_from_row(c_row, base))
        if iq is None or ic is None:
            continue
        ious.append(calculate_iou(iq, ic))
        used_dims.append(base)
    return (float(np.mean(ious)) if ious else 0.0), used_dims


def cat_score_ignore_missing(q_row, c_row):
    """
    Binary equality across categorical fields; ignore NaNs.
    Returns: (S_cat, match_dict, all_equal_bool)
    """
    vals, matches = [], {}
    for f in CAT_FIELDS:
        if f not in q_row or f not in c_row:
            continue
        qa, ca = q_row[f], c_row[f]
        if pd.isna(qa) or pd.isna(ca):
            continue
        m = 1 if qa == ca else 0
        vals.append(m)
        matches[f] = m
    if not vals:
        return 0.5, matches, False  # neutral when nothing to compare
    all_equal = all(v == 1 for v in vals)
    return float(np.mean(vals)), matches, all_equal


def grade_tier_multiplier(q_row, c_row):
    """
    *1.0 exact grade (grade + suffix) match,
    *0.95 same grade_base,
    *0.85 otherwise.
    Returns: (multiplier, exact_grade_bool, same_base_bool)
    """
    qg = ((str(q_row.get("grade")) if pd.notna(q_row.get("grade")) else "") + "|" +
          (str(q_row.get("grade_suffix")) if pd.notna(q_row.get("grade_suffix")) else ""))
    cg = ((str(c_row.get("grade")) if pd.notna(c_row.get("grade")) else "") + "|" +
          (str(c_row.get("grade_suffix")) if pd.notna(c_row.get("grade_suffix")) else ""))
    if qg and cg and qg == cg:
        return GRADE_MULT_EXACT, True, True
    qb, cb = q_row.get("grade_base"), c_row.get("grade_base")
    same_base = (pd.notna(qb) and pd.notna(cb) and qb == cb)
    return (GRADE_MULT_SAME_BASE if same_base else GRADE_MULT_DIFF_BASE), False, same_base


def cp_score_rm_binary(q_row, c_row, tolerance=CP_TOLERANCE):
    """
    CP based only on Tensile strength (Rm)_mid.
    If both present:
      CP = 1 when relative diff <= tolerance, else 0.
    If either missing: CP = 0.30 (fallback).
    """
    qa = pd.to_numeric(q_row.get(RM_COL), errors="coerce")
    ca = pd.to_numeric(c_row.get(RM_COL), errors="coerce")
    if pd.isna(qa) or pd.isna(ca):
        return 0.30
    qa, ca = float(qa), float(ca)
    denom = max(abs(qa), abs(ca))
    if denom == 0.0:
        return 1.0  # both zero => identical
    rel_diff = abs(qa - ca) / denom
    return 1.0 if rel_diff <= tolerance else 0.0


def score_pair(q_row, c_row, allow_category_fallback=False):
    # Category gate / multiplier
    cat_q, cat_c = q_row.get("Category"), c_row.get("Category")
    if pd.isna(cat_q) or pd.isna(cat_c):
        if not allow_category_fallback: 
            return None
        category_multiplier = 0.50
    else:
        if cat_q != cat_c and not allow_category_fallback:
            return None
        category_multiplier = 1.00 if cat_q == cat_c else 0.50

    # Grade tier
    gmult, exact_grade, same_base = grade_tier_multiplier(q_row, c_row)

    # Dimensions & surface
    S_dim, used_dims = dim_score(q_row, c_row)
    S_cat, _matches, _ = cat_score_ignore_missing(q_row, c_row)

    # CP (binary on Rm_mid with 0.30 fallback)
    CP = cp_score_rm_binary(q_row, c_row)

    # Final
    final = category_multiplier * gmult * (0.60*S_dim + 0.30*S_cat + 0.10*CP)

    return {
        "final_score": final,
        "S_dim": S_dim,
        "S_cat": S_cat,
        "CP": CP,
        "grade_mult": gmult,
        "category_mult": category_multiplier,
        "exact_grade": int(exact_grade),
        "same_grade_base": int(same_base),
        "dims_used": used_dims,
    }



def rank_similar_rfqs(joined_df, query_id, top_k=3, allow_category_fallback=False):
    q_row_df = joined_df.loc[joined_df["id"] == query_id]
    if q_row_df.empty:
        raise ValueError(f"RFQ id {query_id} not found.")
    q_row = q_row_df.iloc[0].to_dict()

    # exclude self by id
    if not allow_category_fallback and pd.notna(q_row.get("Category")):
        pool = joined_df[(joined_df["Category"] == q_row["Category"]) & (joined_df["id"] != query_id)]
    else:
        pool = joined_df[joined_df["id"] != query_id]

    A = pool[pool["grade_base"] == q_row.get("grade_base")]
    B = pool[pool["grade_base"] != q_row.get("grade_base")]

    def enrich(frame):
        rows = []
        for _, c in frame.iterrows():
            if c["id"] == query_id:     # safety (should already be excluded)
                continue
            res = score_pair(q_row, c.to_dict(), allow_category_fallback=allow_category_fallback)
            if res is None:
                continue
            rows.append({"query_id": query_id, "candidate_id": c["id"], **res})
        return rows

    rows = enrich(A) + enrich(B)
    def sort_key(r):
        # final desc → S_dim desc → S_cat desc → more dims used
        return (-round(r["final_score"], 12),
                -round(r["S_dim"], 12),
                -round(r["S_cat"], 12),
                -len(r["dims_used"]))

    ranked = sorted(rows, key=sort_key)
    return ranked[:top_k]



def top3_for_all(joined_df, allow_category_fallback=False, top_k=3):
    """
    For each RFQ, compute top-k similar RFQs (exclude same id).
    Returns a tidy DataFrame with only the essentials.
    """
    out_rows = []
    for qid in joined_df["id"].tolist():
        ranked = rank_similar_rfqs(
            joined_df, query_id=qid, top_k=top_k,
            allow_category_fallback=allow_category_fallback
        )
        for rank, r in enumerate(ranked, start=1):
            out_rows.append({
                "query_id": r["query_id"],
                "rank": rank,
                "candidate_id": r["candidate_id"],
                "final_score": round(r["final_score"], 6),
                "S_dim": round(r["S_dim"], 6),
                "S_cat": round(r["S_cat"], 6),
                "CP": r["CP"],
                "dims_used": ", ".join(r["dims_used"]),
            })
    return pd.DataFrame(out_rows)
