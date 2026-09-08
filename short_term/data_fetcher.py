"""
短线数据获取层：日线行情、资金流向、指数数据
复用 core/data_cache.py 的 cached_api_call 缓存机制
"""
import sys
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

from core.data_cache import cached_api_call
from short_term.config import (
    PRICE_CACHE_TTL, FLOW_CACHE_TTL, INDEX_CACHE_TTL,
    MIN_LISTED_DAYS,
)


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_a_stock_list() -> pd.DataFrame:
    """获取全A股列表（含股票代码、名称、流通市值、成交额）"""
    print("[数据] 获取A股全量列表...")
    try:
        # 使用东方财富实时行情获取市值和成交额
        spot_df = cached_api_call(
            ak.stock_zh_a_spot_em,
            max_age_seconds=PRICE_CACHE_TTL,
        )
        if spot_df is not None and len(spot_df) > 0:
            result = []
            for _, row in spot_df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                if not code or len(code) != 6:
                    continue
                result.append({
                    "code": code,
                    "name": str(row.get("名称", "")),
                    "float_mv": safe_float(row.get("流通市值")) or 0,
                    "total_mv": safe_float(row.get("总市值")) or 0,
                    "amount": safe_float(row.get("成交额")) or 0,
                    "turnover_rate": safe_float(row.get("换手率")) or 0,
                    "change_pct": safe_float(row.get("涨跌幅")) or 0,
                })
            stock_df = pd.DataFrame(result)
            print(f"[数据] 全A股共 {len(stock_df)} 只")
            sys.stdout.flush()
            return stock_df
        else:
            print("[数据] 实时行情为空，使用备用方案")
    except Exception as e:
        print(f"[数据] 实时行情获取失败: {e}，使用备用方案")

    # 备用方案：仅代码和名称
    try:
        df = ak.stock_info_a_code_name()
        result = []
        for _, row in df.iterrows():
            result.append({
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name", "")),
                "float_mv": 0,
            })
        stock_df = pd.DataFrame(result)
        print(f"[数据] 全A股共 {len(stock_df)} 只（备用）")
        sys.stdout.flush()
        return stock_df
    except Exception as e2:
        print(f"[数据] 获取股票列表失败: {e2}")
        return pd.DataFrame()


def fetch_daily_kline_batch(codes: list, start_date: str, end_date: str,
                            max_workers: int = 4) -> dict:
    """
    批量获取日线OHLCV数据（东方财富数据源，并发拉取，含重试）

    Args:
        codes: 股票代码列表
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        max_workers: 并发线程数

    Returns:
        dict: {code: DataFrame(columns=[date,open,high,low,close,volume,amount_value,turnover_rate,change_pct])}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time

    print(f"[数据] 批量获取日线 {start_date}-{end_date}，共 {len(codes)} 只（并发:{max_workers}）...")
    result = {}
    errors = [0]
    # v2.1: 东财不可用时跳过重试，直达腾讯源
    eastmoney_down = [False]

    def _fetch_one(code):
        # v2.1: 东财不可用直接跳腾讯，节省时间
        if not eastmoney_down[0]:
            try:
                df = cached_api_call(
                    ak.stock_zh_a_hist,
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust="",
                    max_age_seconds=PRICE_CACHE_TTL,
                )
                if df is not None and len(df) >= 60:
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume",
                        "成交额": "amount_value", "换手率": "turnover_rate",
                        "涨跌幅": "change_pct",
                    })
                    for col in ["open", "high", "low", "close", "volume", "amount_value"]:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                    if "turnover_rate" in df.columns:
                        df["turnover_rate"] = df["turnover_rate"].astype(float)
                    else:
                        df["turnover_rate"] = 0.0
                    if "change_pct" in df.columns:
                        df["change_pct"] = df["change_pct"].astype(float)
                    return (code, df)
                return None
            except Exception:
                # 东财不可用，标记跳过后续重试
                eastmoney_down[0] = True

        # 腾讯数据源
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            df = cached_api_call(
                ak.stock_zh_a_hist_tx,
                symbol=f"{prefix}{code}",
                start_date=start_date, end_date=end_date, adjust="",
                max_age_seconds=PRICE_CACHE_TTL,
            )
            if df is not None and len(df) >= 60:
                df = df.rename(columns={
                    "date": "date", "open": "open", "close": "close",
                    "high": "high", "low": "low",
                })
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                # 腾讯数据源 amount 是成交额（万元），转为元
                if "amount" in df.columns:
                    df["amount_value"] = df["amount"].astype(float) * 10000
                    df["volume"] = df["amount_value"] / (df["close"].astype(float) * 100)
                else:
                    df["volume"] = 0
                df["turnover_rate"] = 0.0
                df["change_pct"] = df["close"].astype(float).pct_change() * 100
                return (code, df)
        except Exception:
            pass

        # v2.2: 东财+腾讯都挂了，走 Sina 日线
        try:
            prefix = "sh" if code.startswith("6") else "sz"
            df = cached_api_call(
                ak.stock_zh_a_daily,
                symbol=f"{prefix}{code}", adjust="",
                max_age_seconds=PRICE_CACHE_TTL,
            )
            if df is not None and len(df) >= 60:
                # Sina 列名: date, open, high, low, close, volume, amount
                # 统一到标准列名
                if "date" not in df.columns and "日期" in df.columns:
                    df = df.rename(columns={"日期": "date"})
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                # amount 列
                if "amount" in df.columns:
                    df["amount_value"] = df["amount"].astype(float)
                elif "成交额" in df.columns:
                    df["amount_value"] = df["成交额"].astype(float)
                else:
                    df["amount_value"] = df["close"].astype(float) * df["volume"].astype(float) * 100
                # 换手率
                if "turnover" in df.columns:
                    df["turnover_rate"] = df["turnover"].astype(float)
                else:
                    df["turnover_rate"] = 0.0
                df["change_pct"] = df["close"].astype(float).pct_change() * 100
                return (code, df)
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, code): code for code in codes}
        with tqdm(total=len(codes), desc="日线拉取", unit="只", ncols=80) as pbar:
            for future in as_completed(futures):
                r = future.result()
                if r is not None:
                    code, df = r
                    result[code] = df
                else:
                    errors[0] += 1
                pbar.set_postfix(valid=len(result), errors=errors[0])
                pbar.update(1)

    print(f"\n[数据] 日线获取完成，有效 {len(result)} 只（{errors[0]} 只失败）")
    sys.stdout.flush()
    return result


def fetch_money_flow_batch(codes: list, date: str) -> dict:
    """
    批量获取个股资金流向

    Args:
        codes: 股票代码列表
        date: 日期 YYYYMMDD

    Returns:
        dict: {code: {main_net_inflow, big_buy, big_sell, big_net_inflow, ...}}
    """
    from tqdm import tqdm

    print(f"[数据] 获取资金流向 {date}，共 {len(codes)} 只...")
    result = {}

    for code in tqdm(codes, desc="资金流向", unit="只", ncols=80):
        try:
            df = cached_api_call(
                ak.stock_individual_fund_flow,
                stock=code, market="sh" if code.startswith("6") else "sz",
                max_age_seconds=FLOW_CACHE_TTL,
            )
            if df is not None and len(df) > 0:
                # 找对应日期的行
                date_str = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                row = df[df["日期"].astype(str).str.contains(date_str.replace("-", ""))]
                if row.empty:
                    row = df.tail(1)  # 取最新

                r = row.iloc[-1]
                result[code] = {
                    "main_net_inflow": safe_float(r.get("主力净流入-净额", r.get("主力净流入-净额", 0))) or 0,
                    "main_net_inflow_rate": safe_float(r.get("主力净流入-净占比", 0)) or 0,
                    "big_net_inflow": safe_float(r.get("超大单净流入-净额", r.get("超大单净流入-净额", 0))) or 0,
                    "big_buy": safe_float(r.get("超大单流入", 0)) or 0,
                    "big_sell": safe_float(r.get("超大单流出", 0)) or 0,
                }
        except Exception:
            pass

    print(f"\n[数据] 资金流向获取完成，有效 {len(result)} 只")
    sys.stdout.flush()
    return result


def fetch_index_daily(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指数日线数据（用于RPS计算）

    Args:
        index_code: 指数代码，如 sh000300
        start_date: 起始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: date, close, change_pct
    """
    print(f"[数据] 获取指数 {index_code} 日线...")
    try:
        df = cached_api_call(
            ak.stock_zh_index_daily_tx,
            symbol=index_code,
            max_age_seconds=INDEX_CACHE_TTL,
        )
        if df is not None and len(df) > 0:
            # 筛选日期范围
            df = df.rename(columns={"date": "date", "close": "close"})
            df["close"] = df["close"].astype(float)
            df = df.sort_values("date")
            # 计算日涨跌幅
            df["change_pct"] = df["close"].pct_change()
            return df
    except Exception as e:
        print(f"[数据] 获取指数失败: {e}")

    return pd.DataFrame()


def fetch_market_cap_float(codes: list) -> dict:
    """获取流通市值映射"""
    print("[数据] 获取流通市值...")
    try:
        df = cached_api_call(
            ak.stock_market_captialization,
            max_age_seconds=PRICE_CACHE_TTL,
        )
        if df is not None and len(df) > 0:
            cap_map = {}
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                cap = safe_float(row.get("流通市值"))
                if code and cap:
                    cap_map[code] = cap
            return cap_map
    except Exception as e:
        print(f"[数据] 获取流通市值失败: {e}")
    return {}
