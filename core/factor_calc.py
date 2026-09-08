import akshare as ak
from config.settings import ROE_MIN, ROE_MAX


def calc_mom60(close_series):
    """计算60日动量"""
    if len(close_series) < 60:
        return None
    return close_series.pct_change(59).iloc[-1]


def get_stock_factors(stock_list, limit=500):
    """批量计算A股全因子：PB/ROE/净利润增速/60日动量"""
    import pandas as pd

    target_codes = stock_list[:limit]

    # === 第1步：批量获取全市场 ROE + 净利润增长率（一次API调用） ===
    print("  批量获取财务数据（全市场 ROE + 利润增速）...")
    try:
        yjbb = ak.stock_yjbb_em(date="20260331")
        codes_arr = yjbb["股票代码"].astype(str).str.zfill(6).values
        roe_arr = yjbb["净资产收益率"].values
        growth_arr = yjbb["净利润-同比增长"].values
        fin_map = dict(zip(codes_arr, zip(roe_arr, growth_arr)))
        import sys
        sys.stdout.flush()
        print(f"  财务数据覆盖 {len(fin_map)} 只股票")
        sys.stdout.flush()
    except Exception as e:
        print(f"  批量财务数据获取失败: {e}")
        fin_map = {}

    # === 第2步：逐只股票取 PB + 价格，结合已缓存的财务数据 ===
    factor_data = []
    total = len(target_codes)

    for idx, code in enumerate(target_codes):
        try:
            fin = fin_map.get(code)
            if fin is None:
                continue
            roe, profit_growth = fin
            if roe is None or profit_growth is None:
                continue
            if roe < ROE_MIN or roe > ROE_MAX:
                continue

            pb_df = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="市净率", period="近一年"
            )
            pb = pb_df["value"].iloc[-1] if not pb_df.empty else None
            if pb is None or pb <= 0:
                continue

            price_df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date="20260201", end_date="20260526",
                adjust="qfq",
            )
            if price_df is None or price_df.empty:
                continue
            mom = calc_mom60(price_df["收盘"])
            if mom is None:
                continue

            factor_data.append([code, pb, roe, profit_growth, mom])

            if (idx + 1) % 50 == 0:
                import sys
                print(f"  A股因子进度: {idx + 1}/{total}（有效: {len(factor_data)}）")
                sys.stdout.flush()
        except Exception:
            continue

    df = pd.DataFrame(
        factor_data,
        columns=["code", "pb", "roe", "profit_growth", "mom_60"],
    )
    return df


def get_hk_stock_factors(hk_stock_df):
    """逐只计算港股因子：PB/ROE/净利润环比增速/60日动量"""
    import pandas as pd

    factor_data = []
    total = len(hk_stock_df)
    print(f"  开始计算港股因子，共 {total} 只标的...")

    for i, (_, row) in enumerate(hk_stock_df.iterrows()):
        code = row["code"]
        try:
            # 财务数据
            fin = ak.stock_hk_financial_indicator_em(symbol=code)
            if fin.empty:
                continue
            pb = fin["市净率"].iloc[0]
            roe = fin["股东权益回报率(%)"].iloc[0]
            profit_growth = fin["净利润滚动环比增长(%)"].iloc[0]
            if pb is None or pb <= 0 or roe is None or profit_growth is None:
                continue
            if roe < ROE_MIN or roe > ROE_MAX:
                continue

            # 价格数据 - 60日动量
            price_df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if price_df is None or price_df.empty:
                continue
            import datetime
            price_df = price_df[price_df["date"] >= datetime.date(2026, 2, 1)]
            if len(price_df) < 60:
                continue
            mom = calc_mom60(price_df["close"])
            if mom is None:
                continue

            factor_data.append([code, pb, roe, profit_growth, mom])

            if (i + 1) % 20 == 0:
                import sys
                print(f"  港股因子进度: {i + 1}/{total}（有效: {len(factor_data)}）")
                sys.stdout.flush()
        except Exception:
            continue

    df = pd.DataFrame(
        factor_data,
        columns=["code", "pb", "roe", "profit_growth", "mom_60"],
    )
    print(f"  港股因子计算完成，有效: {len(df)}/{total}")
    return df


def factor_standardize(df, factor_cols):
    """因子Z-score标准化"""
    for col in factor_cols:
        series = df[col].astype(float)
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            df[f"{col}_norm"] = 0
        else:
            df[f"{col}_norm"] = (series - mean) / std
    return df
