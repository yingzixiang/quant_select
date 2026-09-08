"""
行业Z-Score标准化：按申万一级行业分组，计算行业内Z-Score
小行业（<5只股票）回退到7大板块
"""
import numpy as np
import pandas as pd
from config.macro_config import INDUSTRY_TO_SECTOR


def _map_industry_to_sector(industry_name: str) -> str:
    """申万一级行业 → 7大板块"""
    for key, sector in INDUSTRY_TO_SECTOR.items():
        if key in industry_name:
            return sector
    return "其他"


def industry_zscore_standardize(
    factor_df: pd.DataFrame,
    factor_cols: list,
    factor_directions: dict,
    min_stocks_per_group: int = 5,
) -> pd.DataFrame:
    """
    按行业分组做Z-Score标准化，统一方向（分数越高越优）

    Args:
        factor_df: 含 code, industry, 和各因子列
        factor_cols: 需要标准化的因子列名列表
        factor_directions: {factor_col: "positive"|"negative"}
        min_stocks_per_group: 行业最少股票数，不足则回退到板块

    Returns:
        DataFrame with {col}_norm columns added
    """
    df = factor_df.copy()

    # 构建行业→板块映射
    industry_list = df["industry"].fillna("其他").astype(str).tolist()
    sector_map = {ind: _map_industry_to_sector(ind) for ind in set(industry_list)}
    df["sector"] = [sector_map.get(ind, "其他") for ind in industry_list]

    for col in factor_cols:
        direction = factor_directions.get(col, "positive")
        norm_col = f"{col}_norm"
        df[norm_col] = 0.0

        # 按行业分组
        for industry, group in df.groupby("industry"):
            if len(group) >= min_stocks_per_group:
                _zscore_group(df, group.index, col, norm_col, direction)
            else:
                pass  # 小行业在板块层处理

        # 板块回退：对小行业，使用所属板块的统计量
        for sector, group in df.groupby("sector"):
            small_indices = []
            for idx in group.index:
                ind = df.loc[idx, "industry"]
                ind_count = (df["industry"] == ind).sum()
                if ind_count < min_stocks_per_group:
                    small_indices.append(idx)

            if small_indices and len(small_indices) < len(group):
                # 用板块统计量
                sector_series = df.loc[group.index, col].astype(float)
                mean = sector_series.mean()
                std = sector_series.std(ddof=0)
                if std > 0:
                    for idx in small_indices:
                        val = float(df.loc[idx, col])
                        z = (val - mean) / std
                        df.loc[idx, norm_col] = z if direction == "positive" else -z

        # 最后兜底：全市场Z-Score（对仍然为0的）
        remaining = df[df[norm_col] == 0]
        if len(remaining) > 0:
            series = df[col].astype(float)
            mean = series.mean()
            std = series.std(ddof=0)
            if std > 0:
                for idx in remaining.index:
                    val = float(df.loc[idx, col])
                    z = (val - mean) / std
                    df.loc[idx, norm_col] = z if direction == "positive" else -z

    return df


def _zscore_group(df, indices, col, norm_col, direction):
    """对一组索引计算Z-Score"""
    vals = df.loc[indices, col].astype(float)
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std == 0:
        return
    for idx in indices:
        z = (float(df.loc[idx, col]) - mean) / std
        df.loc[idx, norm_col] = z if direction == "positive" else -z
