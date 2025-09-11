import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# numeric ~ Numeric: Pearson correlation matrix
def numeric_corr(df, numeric_cols=None, method="pearson"):
    """
    Returns a correlation matrix over numeric columns.
    Default method is Pearson; you can pass 'spearman' if needed.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].corr(method=method)


#  Categorical ~ Numeric: Eta-squared (correlation ratio)
def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Eta-squared  measuring how much variance in 'values' is explained by 'categories'.
    Returns a number in [0, 1]. 0 = no association, 1 = perfect separation.
    """
    mask = categories.notna() & values.notna()
    cats = categories[mask].astype("category")
    vals = values[mask].astype(float)

    overall_mean = vals.mean()
    counts = cats.value_counts()
    means = vals.groupby(cats, observed=True).mean()

    ss_between = ((means - overall_mean)**2 * counts.loc[means.index]).sum()
    ss_total = ((vals - overall_mean)**2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def cat_num_assoc_matrix(df, categorical_cols, numeric_cols):
    """
    Returns df where rows = categorical columns, columns = numeric columns,
    and cells = eta-squared (correlation ratio).
    """
    data = {}
    for num in numeric_cols:
        data[num] = [correlation_ratio(df[cat], df[num]) for cat in categorical_cols]
    return pd.DataFrame(data, index=categorical_cols)


# categorical ↔ Categorical: Cramers V (bias-corrected)
def cramers_v(x: pd.Series, y: pd.Series, bias_correction: bool = True) -> float:
    """
    Bias-corrected Cramers V.
    Returns a number in [0, 1]. 0 = no association, 1 = perfect association.
    """
    mask = x.notna() & y.notna()
    table = pd.crosstab(x[mask], y[mask])
    if table.size == 0:
        return 0.0

    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = table.shape

    if bias_correction:
        # Bias correction
        phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
        rcorr = r - (r-1)**2/(n-1)
        kcorr = k - (k-1)**2/(n-1)
        denom = max(1e-12, min(rcorr-1, kcorr-1))
        return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
    else:
        return float(np.sqrt(phi2 / max(1e-12, min(r-1, k-1))))


def cat_cat_assoc_matrix(df, categorical_cols):
    """
    Returns a symmetric df of Cramers V for all pairs of categorical columns.
    """
    cols = list(categorical_cols)
    m = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i: 
                continue
            v = cramers_v(df[a], df[b])
            m.loc[a, b] = v
            m.loc[b, a] = v
    return m