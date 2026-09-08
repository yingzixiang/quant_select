"""
回测模块：A股持仓模拟 + 债券收益 + 多期回测
"""
import numpy as np
import pandas as pd
import akshare as ak


def simulate_holding_return(target_codes, start_date, end_date, stock_weight=1.0):
    """
    A股组合持仓收益 + 债券收益

    Returns:
        (组合净值, 组合收益率%, details_dict)
    """
    stock_ret = 1.0
    if target_codes and stock_weight > 0:
        ret_list = []
        for code in target_codes:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq",
                )
                if df is not None and len(df) >= 2:
                    buy_p = float(df["收盘"].iloc[0])
                    sell_p = float(df["收盘"].iloc[-1])
                    ret_list.append(sell_p / buy_p)
            except Exception:
                continue
        if ret_list:
            stock_ret = float(np.mean(ret_list))

    # 债券部分（简化：用10年国债收益率近似）
    bond_ret = 1.0
    bond_weight = 1.0 - stock_weight
    if bond_weight > 0:
        try:
            rate_df = ak.bond_zh_us_rate()
            cn_10y = float(rate_df["中国国债收益率10年"].dropna().iloc[-1])
            days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            bond_ret = 1.0 + (cn_10y / 100) * (days / 365)
        except Exception:
            pass

    combined_ret = stock_ret * stock_weight + bond_ret * bond_weight
    profit = (combined_ret - 1) * 100

    details = {
        "stock_return": stock_ret,
        "bond_return": bond_ret,
        "combined_return": combined_ret,
        "stock_weight": stock_weight,
        "bond_weight": bond_weight,
    }
    return combined_ret, profit, details


def simulate_hk_holding_return(target_codes, start_date, end_date):
    """
    港股组合持仓收益（等权，纯股票，不掺债）

    使用 ak.stock_hk_daily(前复权) 拉取全历史后按区间裁剪，
    取区间首日收盘为买入价、末日收盘为卖出价，等权平均。

    Returns:
        float: 港股组合净值（1.0 表示不赚不亏）
    """
    ret_list = []
    for code in target_codes:
        try:
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if df is None or len(df) < 2:
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.to_datetime(start_date)) &
                    (df["date"] <= pd.to_datetime(end_date))]
            if len(df) < 2:
                continue
            buy_p = float(df["close"].iloc[0])
            sell_p = float(df["close"].iloc[-1])
            if buy_p > 0:
                ret_list.append(sell_p / buy_p)
        except Exception:
            continue
    if not ret_list:
        return 1.0
    return float(np.mean(ret_list))
