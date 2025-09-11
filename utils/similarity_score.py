import numpy as np
import pandas as pd
import re


DIM_BASES = [
    "thickness", "width", "length", "height",
    "inner_diameter", "outer_diameter", "weight",
]

CAT_FIELDS = ["coating", "finish", "surface_type", "surface_protection", "form"]

RM_COL = "Tensile strength (Rm)_mid"

DEFAULT_CONFIG = {
    "w_dim": 0.60,
    "w_cat": 0.30,
    "w_cp": 0.10,
    "cp_tolerance": 0.10,
    "grade_mult_exact": 1.00,
    "grade_mult_same_base": 0.95,
    "grade_mult_diff_base": 0.85,
    "category_match_mult": 1.00,
    "category_fallback_mult": 0.50,
}


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


def grade_tier_multiplier(q_row, c_row, cfg: dict | None = None):
    """
    *1.0 exact grade (grade + suffix) match,
    *0.95 same grade_base,
    *0.85 otherwise.
    Returns: (multiplier, exact_grade_bool, same_base_bool)
    """
    cfg = (DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **cfg})
    qg = ((str(q_row.get("grade")) if pd.notna(q_row.get("grade")) else "") + "|" +
          (str(q_row.get("grade_suffix")) if pd.notna(q_row.get("grade_suffix")) else ""))
    cg = ((str(c_row.get("grade")) if pd.notna(c_row.get("grade")) else "") + "|" +
          (str(c_row.get("grade_suffix")) if pd.notna(c_row.get("grade_suffix")) else ""))
    if qg and cg and qg == cg:
        return cfg["grade_mult_exact"], True, True
    qb, cb = q_row.get("grade_base"), c_row.get("grade_base")
    same_base = (pd.notna(qb) and pd.notna(cb) and qb == cb)
    return (cfg["grade_mult_same_base"] if same_base else cfg["grade_mult_diff_base"]), False, same_base


def cp_score_rm_binary(q_row, c_row, tolerance: float | None = None, cfg: dict | None = None):
    """
    CP based only on Tensile strength (Rm)_mid.
    If both present:
      CP = 1 when relative diff <= tolerance, else 0.
    If either missing: CP = 0.30 (fallback).
    """
    if tolerance is None:
        cfg = (DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **cfg})
        tolerance = cfg["cp_tolerance"]
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


def score_pair(q_row, c_row, allow_category_fallback: bool = False, cfg: dict | None = None):
    """
    Compute similarity components and final score for a query-candidate pair.
    cfg allows ablations: weights (w_dim, w_cat, w_cp), multipliers, cp_tolerance.
    """
    cfg = (DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **cfg})
    # Category gate / multiplier
    cat_q, cat_c = q_row.get("Category"), c_row.get("Category")
    if pd.isna(cat_q) or pd.isna(cat_c):
        if not allow_category_fallback: 
            return None
        category_multiplier = cfg["category_fallback_mult"]
    else:
        if cat_q != cat_c and not allow_category_fallback:
            return None
        category_multiplier = cfg["category_match_mult"] if cat_q == cat_c else cfg["category_fallback_mult"]

    # Grade tier
    gmult, exact_grade, same_base = grade_tier_multiplier(q_row, c_row, cfg)

    # Dimensions & surface
    S_dim, used_dims = dim_score(q_row, c_row)
    S_cat, _matches, _ = cat_score_ignore_missing(q_row, c_row)

    # CP (binary on Rm_mid with 0.30 fallback)
    CP = cp_score_rm_binary(q_row, c_row, cfg=cfg)

    # Final
    final = category_multiplier * gmult * (cfg["w_dim"]*S_dim + cfg["w_cat"]*S_cat + cfg["w_cp"]*CP)

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



def rank_similar_rfqs(joined_df, query_id, top_k: int = 3, allow_category_fallback: bool = False, cfg: dict | None = None):
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
            res = score_pair(q_row, c.to_dict(), allow_category_fallback=allow_category_fallback, cfg=cfg)
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



def top3_for_all(joined_df, allow_category_fallback: bool = False, top_k: int = 3, cfg: dict | None = None):
    """
    For each RFQ, compute top-k similar RFQs (exclude same id).
    Returns a tidy DataFrame with only the essentials.
    """
    out_rows = []
    for qid in joined_df["id"].tolist():
        ranked = rank_similar_rfqs(
            joined_df, query_id=qid, top_k=top_k,
            allow_category_fallback=allow_category_fallback, cfg=cfg
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


# For Ablation analysis

def run_topk_with_config(joined_df: pd.DataFrame, cfg: dict | None, name: str,
                         top_k: int = 3, allow_category_fallback: bool = True) -> pd.DataFrame:
    df = top3_for_all(joined_df, allow_category_fallback=allow_category_fallback, top_k=top_k, cfg=cfg)
    df = df.copy()
    df["config"] = name
    return df


def compare_topk(base_df: pd.DataFrame, var_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Compare two top-k tables (from run_topk_with_config): per-query metrics.
    Returns per-query rows with:
      - jaccard_k: Jaccard similarity of top-k candidate sets
      - top1_same: 1 if top-1 candidate is the same
      - mean_rank_change: avg |rank_var - rank_base| over intersecting candidates (NaN if none)
      - added: # of candidates in var but not in base
      - removed: # of candidates in base but not in var
    """
    base = base_df[["query_id", "rank", "candidate_id"]].copy()
    var = var_df[["query_id", "rank", "candidate_id"]].copy()

    qids = sorted(set(base["query_id"]).union(set(var["query_id"])) )
    rows = []
    for q in qids:
        b = base[base["query_id"] == q].sort_values("rank").head(k)
        v = var[var["query_id"] == q].sort_values("rank").head(k)
        b_ids = b["candidate_id"].tolist()
        v_ids = v["candidate_id"].tolist()
        inter = set(b_ids).intersection(v_ids)
        union = set(b_ids).union(v_ids)
        jacc = (len(inter) / len(union)) if union else np.nan
        top1_same = int(len(b_ids) > 0 and len(v_ids) > 0 and b_ids[0] == v_ids[0])
        # rank maps
        b_ranks = {cid: int(rk) for cid, rk in zip(b["candidate_id"], b["rank"]) }
        v_ranks = {cid: int(rk) for cid, rk in zip(v["candidate_id"], v["rank"]) }
        if inter:
            mean_rank_change = float(np.mean([abs(b_ranks[c]-v_ranks[c]) for c in inter]))
        else:
            mean_rank_change = np.nan
        rows.append({
            "query_id": q,
            "jaccard_k": jacc,
            "top1_same": top1_same,
            "mean_rank_change": mean_rank_change,
            "added": len(set(v_ids) - set(b_ids)),
            "removed": len(set(b_ids) - set(v_ids)),
        })
    return pd.DataFrame(rows)
