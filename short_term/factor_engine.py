"""
因子预处理引擎：缺失值填充、标准化、缩尾、IC筛选
"""
import numpy as np
import pandas as pd
import sys
from scipy import stats


def fill_missing(df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """
    缺失值填充

    Args:
        df: index=code, columns=factors
        method: "median" | "zero" | "mean"

    Returns:
        填充后的DataFrame
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if df[col].isna().sum() == len(df):
            df[col] = 0.0
            continue

        if method == "median":
            fill_val = df[col].median()
        elif method == "mean":
            fill_val = df[col].mean()
        else:
            fill_val = 0.0

        if pd.isna(fill_val):
            fill_val = 0.0

        df[col] = df[col].fillna(fill_val)

    return df


def winsorize(df: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    极值缩尾：将超出分位数的值截断到分位边界

    Args:
        df: factor DataFrame
        lower_pct: 下分位数（默认1%）
        upper_pct: 上分位数（默认99%）

    Returns:
        缩尾后的DataFrame
    """
    df = df.copy()
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        lo = series.quantile(lower_pct)
        hi = series.quantile(upper_pct)
        if lo == hi:
            continue
        df[col] = df[col].clip(lo, hi)
    return df


def zscore_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面Z-Score标准化：(x - mean) / std

    Args:
        df: factor DataFrame

    Returns:
        标准化后的DataFrame
    """
    df = df.copy()
    for col in df.columns:
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if std > 0:
            df[col] = (series - mean) / std
        else:
            df[col] = 0.0
    return df


def calculate_ic(factor_df: pd.DataFrame, forward_returns: pd.Series, method: str = "rank") -> pd.Series:
    """
    计算各因子的IC（Information Coefficient）

    Args:
        factor_df: index=code, columns=factors
        forward_returns: index=code, 未来N日收益率
        method: "rank" (Rank IC) | "pearson" (Pearson IC)

    Returns:
        Series: index=factor_name, value=IC
    """
    common_codes = factor_df.index.intersection(forward_returns.index)
    if len(common_codes) < 30:
        print("[因子引擎] 公共样本不足，无法计算IC")
        return pd.Series(dtype=float)

    ic_dict = {}
    for col in factor_df.columns:
        factor_vals = factor_df.loc[common_codes, col]
        ret_vals = forward_returns.loc[common_codes]

        valid = factor_vals.notna() & ret_vals.notna()
        if valid.sum() < 30:
            ic_dict[col] = 0.0
            continue

        f = factor_vals[valid]
        r = ret_vals[valid]

        try:
            if method == "rank":
                ic, _ = stats.spearmanr(f, r)
            else:
                ic, _ = stats.pearsonr(f, r)
            ic_dict[col] = ic if not np.isnan(ic) else 0.0
        except Exception:
            ic_dict[col] = 0.0

    return pd.Series(ic_dict).sort_values(ascending=False)


def filter_by_ic(factor_df: pd.DataFrame, ic_series: pd.Series,
                 min_abs_ic: float = 0.03, max_factors: int = 80) -> pd.DataFrame:
    """
    根据IC筛选因子：保留 |IC| > min_abs_ic 的因子，最多 max_factors 个

    Args:
        factor_df: 原始因子DataFrame
        ic_series: IC值Series
        min_abs_ic: 最小绝对IC阈值
        max_factors: 最大保留因子数

    Returns:
        筛选后的因子DataFrame
    """
    # 按绝对IC降序排序
    ic_abs = ic_series.abs().sort_values(ascending=False)

    # 筛选
    selected = []
    for factor in ic_abs.index:
        if ic_abs[factor] >= min_abs_ic and factor in factor_df.columns:
            selected.append(factor)
        if len(selected) >= max_factors:
            break

    print(f"[因子引擎] IC筛选：{len(factor_df.columns)} → {len(selected)} 个因子（min_abs_ic={min_abs_ic}）")
    sys.stdout.flush()
    return factor_df[selected]


def preprocess_factors(factor_df: pd.DataFrame, forward_returns: pd.Series = None,
                       min_abs_ic: float = 0.03, max_factors: int = 80) -> pd.DataFrame:
    """
    因子预处理完整管线：填充 → 缩尾 → 标准化 → (可选IC筛选)

    Args:
        factor_df: index=code, columns=factors
        forward_returns: 可选，未来收益用于IC筛选
        min_abs_ic: IC筛选阈值
        max_factors: 最大保留因子数

    Returns:
        预处理后的因子DataFrame
    """
    print(f"[因子引擎] 预处理开始，原始因子: {len(factor_df.columns)} 个，样本: {len(factor_df)} 只")
    sys.stdout.flush()

    # 1. 缺失值填充
    df = fill_missing(factor_df, method="median")
    print(f"[因子引擎] Step1 缺失值填充完成")

    # 2. 极值缩尾
    df = winsorize(df)
    print(f"[因子引擎] Step2 极值缩尾完成")

    # 3. Z-Score标准化
    df = zscore_standardize(df)
    print(f"[因子引擎] Step3 Z-Score标准化完成")

    # 4. IC筛选（如果有未来收益数据）
    if forward_returns is not None and len(forward_returns) > 0:
        ic = calculate_ic(df, forward_returns)
        df = filter_by_ic(df, ic, min_abs_ic=min_abs_ic, max_factors=max_factors)

    # 5. 再次填充和标准化（因子筛选后可能有新的缺失）
    df = fill_missing(df, method="median")
    df = zscore_standardize(df)

    # 6. 去除全0或常数列
    valid_cols = [c for c in df.columns if df[c].std() > 1e-8]
    df = df[valid_cols]

    print(f"[因子引擎] 预处理完成，最终: {len(df.columns)} 个因子，{len(df)} 只样本")
    sys.stdout.flush()
    return df


def compute_factor_composite_score(factor_df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """
    计算因子综合得分：等权或加权求和

    Args:
        factor_df: 预处理后的因子DataFrame
        weights: 因子权重字典 {factor_name: weight}，None则等权

    Returns:
        Series: index=code, value=综合得分
    """
    if factor_df.empty:
        return pd.Series(dtype=float)

    if weights is None:
        # 等权：IC越高的因子隐式等权（已标准化）
        weights = {col: 1.0 / len(factor_df.columns) for col in factor_df.columns}

    score = pd.Series(0.0, index=factor_df.index)
    for col in factor_df.columns:
        if col in weights:
            score += factor_df[col] * weights[col]

    return score
