"""
intraday/data_fetcher.py
分钟级数据获取层：个股 K 线、指数、板块、成交量剖面

复用 core.data_cache.cached_api_call 缓存机制。
数据源：AKShare (东方财富为主，新浪备用)
"""
import sys
import time
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from core.data_cache import cached_api_call
from intraday.config import DATA_CACHE_TTL, DATA_DEGRADATION

# ---- 全局降级日志 ----
_warnings_issued: set = set()


def _warn_once(key: str, message: str):
    """同类降级警告只打印一次"""
    if key not in _warnings_issued:
        print(f"[数据] ⚠ {message}")
        _warnings_issued.add(key)


def safe_float(val) -> Optional[float]:
    """安全转换为 float，失败返回 None"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ============================================================
# 个股分钟 K 线
# ============================================================

def fetch_minute_kline(symbol: str,
                       start_date: str,
                       end_date: str,
                       period: str = "5") -> Optional[pd.DataFrame]:
    """
    获取单只个股分钟级 K 线数据（东方财富数据源）

    Args:
        symbol: 股票代码，如 "000001"
        start_date: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
        end_date: 结束日期
        period: K 线周期 "1"/"5"/"15"/"30"/"60"

    Returns:
        DataFrame with columns:
            timestamp(datetime), open, high, low, close, volume, amount, vwap
        失败返回 None
    """
    # 统一日期格式
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    if len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    for attempt in range(3):
        try:
            df = cached_api_call(
                ak.stock_zh_a_hist_min_em,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                period=period,
                adjust="",
                max_age_seconds=DATA_CACHE_TTL["minute_kline"],
            )

            if df is not None and len(df) >= 10:
                return _normalize_minute_df(df, symbol)
            return None

        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
            else:
                # 东方财富失败，尝试新浪备用（仅 period="1" 时可用）
                if period == "1":
                    return _fetch_minute_kline_sina_fallback(symbol)
                return None


def _normalize_minute_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """将 AKShare 返回的分钟数据标准化"""
    # 列名映射
    col_map = {
        "时间": "timestamp",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "均价": "vwap",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 如果没有 amount，从 volume * vwap 估算
    if "amount" not in df.columns and "volume" in df.columns and "vwap" in df.columns:
        df["amount"] = df["volume"] * df["vwap"] * 100
    elif "amount" not in df.columns and "volume" in df.columns and "close" in df.columns:
        df["amount"] = df["volume"] * df["close"] * 100

    # 如果没有 vwap，用 (H+L+C)/3 近似
    if "vwap" not in df.columns:
        if all(c in df.columns for c in ["high", "low", "close"]):
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        else:
            df["vwap"] = df["close"]

    # 解析时间戳
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def _fetch_minute_kline_sina_fallback(symbol: str) -> Optional[pd.DataFrame]:
    """新浪备用数据源（仅 1 分钟 K 线，最多 ~1970 bar）"""
    try:
        prefix = "sh" if symbol.startswith("6") else "sz"
        full_symbol = f"{prefix}{symbol}"

        df = cached_api_call(
            ak.stock_zh_a_minute,
            symbol=full_symbol,
            period="1",
            max_age_seconds=DATA_CACHE_TTL["minute_kline"],
        )

        if df is not None and len(df) >= 10:
            df = df.rename(columns={
                "day": "date_str",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            # 合成 timestamp
            if "date_str" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date_str"])
            df["amount"] = df["volume"] * df["close"] * 100
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
            return df

    except Exception as e:
        _warn_once("sina_minute_fallback", f"新浪备用数据源失败: {e}")

    return None


# ============================================================
# 指数分钟 K 线
# ============================================================

def fetch_index_minute(symbol: str = "000300",
                       start_date: str = None,
                       end_date: str = None,
                       period: str = "5") -> Optional[pd.DataFrame]:
    """
    获取指数分钟级 K 线数据

    Args:
        symbol: 指数代码，默认 "000300"（沪深300）
        start_date: 起始日期
        end_date: 结束日期
        period: K 线周期

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, amount
    """
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    if len(end_date) == 8:
        end_date_8 = end_date
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
    else:
        end_date_8 = end_date.replace("-", "")

    for attempt in range(2):
        try:
            df = cached_api_call(
                ak.index_zh_a_hist_min_em,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                max_age_seconds=DATA_CACHE_TTL["index_minute"],
            )

            if df is not None and len(df) >= 10:
                col_map = {
                    "时间": "timestamp",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                return df

        except Exception as e:
            if attempt < 1:
                time.sleep(2.0)
            else:
                print(f"[数据] 获取指数 {symbol} 分钟数据失败: {e}")

    return None


# ============================================================
# 板块分钟数据（优雅降级）
# ============================================================

def fetch_sector_minute(sector_name: str,
                        period: str = "5") -> Optional[pd.DataFrame]:
    """
    获取板块分钟 K 线（东方财富板块数据）

    Args:
        sector_name: 板块名称，如 "白酒"、"半导体"
        period: K 线周期

    Returns:
        DataFrame 或 None（数据不可用时优雅降级）
    """
    if not DATA_DEGRADATION["sector_minute_available"]:
        return None

    try:
        df = cached_api_call(
            ak.stock_board_industry_hist_min_em,
            symbol=sector_name,
            period=period,
            max_age_seconds=DATA_CACHE_TTL["sector_minute"],
        )

        if df is not None and len(df) >= 5:
            col_map = {
                "时间": "timestamp",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    except Exception as e:
        _warn_once(f"sector_minute_{sector_name}", f"板块 {sector_name} 分钟数据不可用: {e}")

    return None


def fetch_sector_spot() -> Optional[pd.DataFrame]:
    """
    获取板块实时行情快照（用于涨跌家数等）

    Returns:
        DataFrame 或 None
    """
    try:
        df = cached_api_call(
            ak.stock_board_industry_name_em,
            max_age_seconds=DATA_CACHE_TTL["sector_spot"],
        )

        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        _warn_once("sector_spot", f"板块快照数据不可用: {e}")

    return None


# ============================================================
# 板块-个股映射
# ============================================================

def build_stock_sector_map() -> Dict[str, str]:
    """
    构建 {股票代码: 所属板块名称} 映射

    使用东方财富行业板块成分股接口构建映射。
    只执行一次并缓存结果。

    Returns:
        dict: {code: sector_name}
    """
    try:
        # 获取板块列表
        sector_df = cached_api_call(
            ak.stock_board_industry_name_em,
            max_age_seconds=DATA_CACHE_TTL["sector_map"],
        )

        if sector_df is None or sector_df.empty:
            _warn_once("sector_map", "板块列表为空，板块过滤将不可用")
            return {}

        sector_names = sector_df["板块名称"].tolist()
        stock_sector: Dict[str, str] = {}

        for i, sname in enumerate(sector_names):
            try:
                cons_df = cached_api_call(
                    ak.stock_board_industry_cons_em,
                    symbol=sname,
                    max_age_seconds=DATA_CACHE_TTL["sector_map"],
                )
                if cons_df is not None and not cons_df.empty:
                    code_col = None
                    for c in ["代码", "code"]:
                        if c in cons_df.columns:
                            code_col = c
                            break
                    if code_col:
                        for code in cons_df[code_col].astype(str):
                            stock_sector[code.zfill(6)] = sname
            except Exception:
                continue

            if (i + 1) % 20 == 0:
                print(f"[数据] 板块映射进度: {i+1}/{len(sector_names)}")

        print(f"[数据] 板块映射完成: {len(stock_sector)} 只个股 → {len(sector_names)} 个板块")
        return stock_sector

    except Exception as e:
        _warn_once("sector_map_build", f"构建板块映射失败: {e}")
        return {}


# ============================================================
# 成交量剖面（历史同时段均量）
# ============================================================

def build_volume_profile(minute_dict: Dict[str, pd.DataFrame],
                         lookback_days: int = 5) -> Dict[Tuple[str, str], float]:
    """
    预计算每只股票在每个 5 分钟时段的近 N 日平均成交量

    Args:
        minute_dict: {code: minute_kline_df}
        lookback_days: 回看天数

    Returns:
        dict: {(code, time_str): avg_volume}
            其中 time_str 格式为 "HH:MM"（如 "09:40"）
    """
    profile: Dict[Tuple[str, str], float] = {}

    for code, df in minute_dict.items():
        if df is None or len(df) < 10:
            continue

        if "timestamp" not in df.columns:
            continue

        # 提取时分
        time_str = df["timestamp"].dt.strftime("%H:%M")
        df_temp = df.assign(time_key=time_str)

        # 按日期分组，取最近 lookback_days 天
        df_temp["date"] = df["timestamp"].dt.date
        recent_dates = sorted(df_temp["date"].unique())[-lookback_days:]

        # 计算每个时段在近 N 日的平均 volume
        for tk in df_temp["time_key"].unique():
            mask = (df_temp["time_key"] == tk) & (df_temp["date"].isin(recent_dates))
            avg_vol = df_temp.loc[mask, "volume"].mean()
            if not np.isnan(avg_vol) and avg_vol > 0:
                profile[(code, tk)] = avg_vol

    return profile


# ============================================================
# 批量获取日线（用于标的池初筛）
# ============================================================

def fetch_daily_kline_for_pool(codes: List[str],
                               start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
    """
    获取日线数据，用于标的池初筛（计算日均成交额、振幅等）

    复用 short_term 的日线获取逻辑。

    Args:
        codes: 股票代码列表
        start_date: YYYYMMDD
        end_date: YYYYMMDD

    Returns:
        {code: kline_df}
    """
    # 导入现有模块
    from short_term.data_fetcher import fetch_daily_kline_batch
    return fetch_daily_kline_batch(codes, start_date, end_date, max_workers=4)


# ============================================================
# 获取 A 股全量列表
# ============================================================

def fetch_a_stock_list() -> pd.DataFrame:
    """获取全 A 股列表（复用短线模块）"""
    from short_term.data_fetcher import fetch_a_stock_list as _fetch
    return _fetch()


# ============================================================
# 数据预加载：一次性拉取回测所需全部分钟数据
# ============================================================

def preload_backtest_data(stock_codes: List[str],
                          start_date: str,
                          end_date: str,
                          period: str = "5") -> Dict:
    """
    预加载回测所需的全部分钟数据

    Args:
        stock_codes: 标的池股票代码
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        period: K 线周期

    Returns:
        {
            "stock_minute": {code: df},
            "index_minute": df,
            "sector_minute": {sector_name: df},
            "sector_map": {code: sector_name},
            "volume_profile": {(code, time): avg_vol},
        }
    """
    print(f"\n[数据] ===== 预加载回测数据 =====")
    print(f"[数据] 标的: {len(stock_codes)} 只 | 区间: {start_date}~{end_date} | K线: {period}min")
    sys.stdout.flush()

    # 1. 指数分钟数据
    print("[数据] 1/5 加载指数分钟数据...")
    index_df = fetch_index_minute("000300", start_date, end_date, period)
    if index_df is not None:
        print(f"[数据]   沪深300: {len(index_df)} 根 {period}min K 线")
    else:
        print("[数据]   ⚠ 沪深300 指数数据加载失败")
    sys.stdout.flush()

    # 2. 个股分钟数据
    print("[数据] 2/5 加载个股分钟数据...")
    stock_dict = {}
    for i, code in enumerate(stock_codes):
        df = fetch_minute_kline(code, start_date, end_date, period)
        if df is not None and len(df) >= 20:
            stock_dict[code] = df
            if (i + 1) % 2 == 0:
                print(f"[数据]   进度: {i+1}/{len(stock_codes)} ({code} ✓)")
        else:
            print(f"[数据]   ⚠ {code} 分钟数据不可用")
        sys.stdout.flush()

    print(f"[数据]   有效个股: {len(stock_dict)}/{len(stock_codes)}")
    sys.stdout.flush()

    # 3. 板块映射
    print("[数据] 3/5 构建板块映射...")
    sector_map = build_stock_sector_map()
    sys.stdout.flush()

    # 4. 板块分钟数据（仅加载相关板块）
    print("[数据] 4/5 加载板块分钟数据...")
    sector_dict = {}
    relevant_sectors = set()
    for code in stock_dict:
        sec = sector_map.get(code, "")
        if sec:
            relevant_sectors.add(sec)

    for i, sec in enumerate(list(relevant_sectors)[:20]):  # 最多 20 个板块
        sec_df = fetch_sector_minute(sec, period)
        if sec_df is not None:
            sector_dict[sec] = sec_df
    print(f"[数据]   板块数据: {len(sector_dict)}/{len(relevant_sectors)} 个")
    sys.stdout.flush()

    # 5. 成交量剖面
    print("[数据] 5/5 构建成交量剖面...")
    vol_profile = build_volume_profile(stock_dict)
    print(f"[数据]   成交量剖面: {len(vol_profile)} 条")

    print(f"[数据] ===== 数据预加载完成 =====\n")
    sys.stdout.flush()

    return {
        "stock_minute": stock_dict,
        "index_minute": index_df,
        "sector_minute": sector_dict,
        "sector_map": sector_map,
        "volume_profile": vol_profile,
    }
