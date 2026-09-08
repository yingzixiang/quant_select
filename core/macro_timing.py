"""
宏观择时模块：18项指标打分 → 宏观总分 → 股债配比 + 风格判定
"""
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import akshare as ak
from config.settings import (
    MACRO_INDICATOR_WEIGHTS, MANUAL_MACRO_DEFAULTS,
    STOCK_ALLOC_FLOOR, STYLE_LOOKBACK_DAYS, STYLE_THRESHOLD, STYLE_INDICES,
)
from config.macro_config import MACRO_INDICATORS
from config.manual_macro import MANUAL_MACRO
from core.data_cache import cached_api_call

# 数据新鲜度告警阈值（天），超过此天数未更新视为过期
STALE_WARNING_DAYS = 90


def _get_api_func(func_name: str):
    """从akshare模块获取API函数"""
    if func_name is None:
        return None
    return getattr(ak, func_name, None)


def _extract_value(result, cfg: dict) -> float:
    """从API返回中提取数值"""
    value_key = cfg.get("value_key")
    if value_key is None:
        return None
    if isinstance(result, dict):
        return result.get(value_key)
    if hasattr(result, "iloc"):
        if value_key not in result.columns:
            return None
        series = result[value_key].dropna()
        if len(series) == 0:
            return None
        # newest_first=True: 数据从新到旧排列，取第一条
        # 默认(oldest_first): 数据从旧到新排列，取最后一条
        if cfg.get("newest_first"):
            return float(series.iloc[0])
        return float(series.iloc[-1])
    return None


def _score_threshold(value: float, direction: str, thresholds: dict) -> float:
    """分段阈值打分 → 归一化到 [0, 1]"""
    if value is None or np.isnan(value):
        return 0.5  # 缺失值给中性分
    breakpoints = sorted(thresholds.items(), key=lambda x: x[0], reverse=(direction == "positive"))
    for bp, score in breakpoints:
        if direction == "positive":
            if value >= bp:
                return score / 10.0
        elif direction == "negative":
            if value <= bp:
                return score / 10.0
    return 0.2  # 最低分


def _score_center(value: float, center_ranges: list) -> float:
    """区间最优打分 → 归一化到 [0, 1]"""
    if value is None or np.isnan(value):
        return 0.5
    for (lo, hi), score in center_ranges:
        if lo <= value < hi:
            return score / 10.0
    return 0.2


def _check_data_freshness(result, cfg: dict) -> dict:
    """检查数据新鲜度，返回 {'stale': bool, 'latest_date': str, 'days_behind': int}"""
    date_key = cfg.get("date_key")
    if not date_key or not hasattr(result, "iloc"):
        return {"stale": False, "latest_date": None, "days_behind": 0}
    if date_key not in result.columns:
        return {"stale": False, "latest_date": None, "days_behind": 0}

    date_series = result[date_key].dropna()
    if len(date_series) == 0:
        return {"stale": False, "latest_date": None, "days_behind": 0}

    # 取最新日期（根据排序方向）
    date_str = str(date_series.iloc[0] if cfg.get("newest_first") else date_series.iloc[-1])
    date_format = cfg.get("date_format", "")

    parsed_date = None
    if date_format == "cn_month":
        # 格式: "2026年04月份"
        m = re.match(r"(\d{4})年(\d{2})月份?", date_str)
        if m:
            parsed_date = datetime(int(m.group(1)), int(m.group(2)), 1)
    elif date_format == "iso_date":
        # 格式: "2026-04-01"
        try:
            parsed_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            pass

    if parsed_date is None:
        return {"stale": False, "latest_date": date_str, "days_behind": 0}

    days_behind = (datetime.now() - parsed_date).days
    stale = days_behind > STALE_WARNING_DAYS
    return {"stale": stale, "latest_date": date_str, "days_behind": days_behind}


def _score_percentile(value: float, history: list, direction: str, percentile_map: dict) -> float:
    """分位打分 → 归一化到 [0, 1]"""
    if value is None or np.isnan(value) or len(history) == 0:
        return 0.5
    pct = (np.searchsorted(np.sort(history), value) / len(history))
    if direction == "negative":
        pct = 1.0 - pct
    for bp, score in sorted(percentile_map.items(), reverse=True):
        if pct >= bp:
            return score / 10.0
    return 0.2


def fetch_all_macro_indicators(manual_overrides: dict = None) -> dict:
    """
    拉取全部18项宏观指标并打分

    Args:
        manual_overrides: {indicator_name: 2-10分} 覆盖手动输入指标

    Returns:
        {indicator_name: {"raw_value": ..., "score": ..., "available": bool}}
    """
    overrides = manual_overrides or {}
    results = {}

    # 合并手动默认值（新 config/manual_macro.py 优先于旧 settings.py）
    manual_defaults = {**MANUAL_MACRO_DEFAULTS}
    for name, val in MANUAL_MACRO.items():
        if val is not None:
            manual_defaults[name] = val

    for name, cfg in MACRO_INDICATORS.items():
        try:
            # 优先级: CLI --manual-macro > config/manual_macro.py > API
            if name in overrides:
                manual_val = overrides[name]
                results[name] = {"raw_value": manual_val, "score": manual_val / 10.0, "available": True, "manual": True}
                continue

            # config/manual_macro.py 中设置的非 None 值覆盖 API
            if name in manual_defaults:
                manual_val = manual_defaults[name]
                results[name] = {"raw_value": manual_val, "score": manual_val / 10.0, "available": True, "manual": True}
                continue

            scoring_type = cfg.get("scoring_type", "threshold")

            # 代理指标（如消费者信心 = PMI + 工业增加值均值）
            if scoring_type == "proxy":
                proxy_sources = cfg.get("proxy_source", [])
                proxy_scores = [results[s]["score"] for s in proxy_sources if s in results]
                proxy_val = np.mean(proxy_scores) if proxy_scores else 0.5
                results[name] = {"raw_value": None, "score": proxy_val, "available": True, "proxy": True}
                continue

            # 特殊处理：先提取 compute_type
            compute_type = cfg.get("compute")

            # API指标：没有 api_func 但有 compute_type 的走纯计算逻辑
            func_name = cfg.get("api_func")
            func = _get_api_func(func_name)

            if func is None and compute_type is None:
                # 没有API也没有特殊计算的指标（如需要手动输入的指标）
                results[name] = {"raw_value": None, "score": 0.5, "available": False}
                continue

            # 有 api_func 的先拉数据，纯 compute 的跳过此步
            result = None
            if func is not None:
                result = cached_api_call(func, max_age_seconds=86400, **cfg.get("api_kwargs", {}))
                if result is None or (hasattr(result, "empty") and result.empty):
                    results[name] = {"raw_value": None, "score": 0.5, "available": False}
                    continue

            if compute_type == "cn_minus_us":
                # 中美利差：取各自最后有效值
                cn_series = result["中国国债收益率10年"].dropna()
                us_series = result["美国国债收益率10年"].dropna()
                if len(cn_series) == 0 or len(us_series) == 0:
                    value = None
                else:
                    cn_10y = float(cn_series.iloc[-1])
                    us_10y = float(us_series.iloc[-1])
                    value = cn_10y - us_10y
            elif compute_type == "turnover_ratio":
                # 量比 = 今日成交额 / 近20日均成交额（上证综指）
                col = cfg["value_key"]
                if col in result.columns:
                    vals = result[col].dropna().values.astype(float)
                    if len(vals) >= 21:
                        today_amt = vals[-1]
                        avg_20 = np.mean(vals[-21:-1])
                        value = today_amt / avg_20 if avg_20 > 0 else 1.0
                    elif len(vals) >= 2:
                        value = float(vals[-1]) / float(vals[-2])
                    else:
                        value = None
                else:
                    value = None
            elif compute_type == "advance_decline_20d":
                # 近20日收涨天数占比（上证综指）
                col = cfg["value_key"]
                if col in result.columns:
                    closes = result[col].dropna().values.astype(float)
                    if len(closes) >= 21:
                        recent = closes[-21:]
                        up_days = sum(1 for i in range(1, len(recent))
                                      if recent[i] > recent[i - 1])
                        value = up_days / 20.0
                    elif len(closes) >= 2:
                        up_days = sum(1 for i in range(1, len(closes))
                                      if closes[i] > closes[i - 1])
                        value = up_days / (len(closes) - 1)
                    else:
                        value = None
                else:
                    value = None
            elif compute_type == "erp":
                # 权益风险溢价 = (100 / PE) - 10年期国债收益率
                pe_value = _extract_value(result, cfg)
                # 从已计算的 bond_10y 获取国债收益率
                bond_info = results.get("bond_10y", {})
                bond_yield = None
                if bond_info.get("available") and isinstance(bond_info.get("raw_value"), (int, float)):
                    bond_yield = bond_info["raw_value"]
                else:
                    # 独立获取国债收益率
                    try:
                        bond_result = cached_api_call(ak.bond_zh_us_rate, max_age_seconds=86400)
                        if bond_result is not None and "中国国债收益率10年" in bond_result.columns:
                            bond_ys = bond_result["中国国债收益率10年"].dropna()
                            if len(bond_ys) > 0:
                                bond_yield = float(bond_ys.iloc[-1])
                    except Exception:
                        pass
                    if bond_yield is None:
                        bond_yield = 2.8
                if pe_value is not None and pe_value > 0:
                    value = (100.0 / pe_value) - bond_yield
                else:
                    value = None
            elif compute_type in ("size_pe_pct", "growth_pe_pct", "pe_spread"):
                # 风格PE估值分位：独立获取上证/创业板PE历史，计算分位和价差
                value = _compute_style_pe(compute_type)
            elif cfg.get("fx_filter"):
                # 汇率过滤特定货币对
                if "货币对" in result.columns:
                    filtered = result[result["货币对"] == cfg["fx_filter"]]
                    if not filtered.empty:
                        value = float(filtered[cfg["value_key"]].iloc[0])
                    else:
                        value = None
                else:
                    value = None
            elif cfg.get("compute_yoy"):
                # 计算同比（如社融）
                col = cfg["value_key"]
                if col in result.columns:
                    vals = result[col].dropna().values
                    if len(vals) >= 13:  # 至少13个月才能同比
                        # 当前月 vs 去年同期（取最近12个月均值的同比变化）
                        recent = vals[-1]
                        year_ago = vals[-13]
                        value = ((recent - year_ago) / abs(year_ago)) * 100 if year_ago != 0 else 0
                    else:
                        value = float(vals[-1])
                else:
                    value = None
            elif cfg.get("compute_mom"):
                # 计算环比增速（如两融余额）
                col = cfg["value_key"]
                if col in result.columns:
                    vals = result[col].dropna().values
                    # 合并沪深两市
                    if "dual_source" in cfg:
                        dual_func = getattr(ak, cfg["dual_source"], None)
                        if dual_func:
                            dual_result = cached_api_call(dual_func, max_age_seconds=86400)
                            if dual_result is not None and not dual_result.empty:
                                dual_vals = dual_result[col].dropna().values
                                vals = vals[-min(len(vals), len(dual_vals)):] + dual_vals[-min(len(vals), len(dual_vals)):]
                    if len(vals) >= 2:
                        value = ((vals[-1] - vals[-2]) / abs(vals[-2])) * 100
                    else:
                        value = float(vals[-1])
                else:
                    value = None
            else:
                value = _extract_value(result, cfg)

            # 打分
            if scoring_type == "percentile":
                # 构建历史序列
                if cfg.get("aggregate") == "sum" and cfg["value_key"] in result.columns:
                    total = float(result[cfg["value_key"]].sum())
                    # 简化：用当前值相对于自身做分位估算
                    score = _score_percentile(total, [total], "positive", cfg.get("percentile_map", {}))
                    value = total
                elif cfg.get("aggregate") == "monthly_sum":
                    col = cfg["value_key"]
                    if col in result.columns:
                        value = float(result[col].tail(21).sum())  # 近月约21交易日
                        lookback_vals = result[col].rolling(21).sum().dropna().values
                        score = _score_percentile(value, lookback_vals, "positive", cfg.get("percentile_map", {}))
                    else:
                        value = None
                        score = 0.5
                else:
                    score = _score_percentile(value, [value], "positive", cfg.get("percentile_map", {}))
            elif scoring_type == "center":
                score = _score_center(value, cfg.get("center_ranges", []))
            else:
                score = _score_threshold(value, cfg["direction"], cfg.get("thresholds", {}))

            results[name] = {"raw_value": value, "score": score, "available": True}

            # 数据新鲜度检查
            freshness = _check_data_freshness(result, cfg)
            if freshness["stale"]:
                results[name]["stale"] = True
                results[name]["stale_info"] = f"最新数据日期 {freshness['latest_date']}，已滞后 {freshness['days_behind']} 天"
            elif freshness["latest_date"]:
                results[name]["latest_date"] = freshness["latest_date"]

        except Exception as e:
            results[name] = {"raw_value": None, "score": 0.5, "available": False, "error": str(e)}

    return results


def compute_macro_score(indicator_scores: dict) -> float:
    """加权计算宏观总分 [0, 1]"""
    total_weight = 0.0
    weighted_sum = 0.0
    for name, weight in MACRO_INDICATOR_WEIGHTS.items():
        info = indicator_scores.get(name, {})
        if info.get("available", False):
            weighted_sum += info["score"] * weight
            total_weight += weight
    if total_weight == 0:
        return 0.5
    return weighted_sum / total_weight  # 归一化到可用权重


def derive_allocation(macro_score: float, indicator_scores: dict = None,
                     month_label: str = None) -> tuple:
    """
    股票仓位 = FLOOR + (1-FLOOR) × 宏观总分
    债券仓位 = 1 - 股票仓位

    逆向修正（分阶段缓慢调整）：
      - 极度悲观（宏观<0.35）且 ERP 高/涨跌比冰点 → 首月仅观察警告
      - 连续 ≥2 个月悲观确认 → 每月 +2%，第8个月到顶 +15%
      - 极度乐观（宏观>0.65）且量比过热/PE泡沫 → 首月观察警告
      - 连续 ≥2 个月过热确认 → 每月 -2%，第7个月到顶 -12%
      - 环境回暖（退出极端区间）→ 立即重置计数
    """
    from core.contrarian_state import compute_contrarian_correction

    indicator_scores = indicator_scores or {}
    stock_weight = STOCK_ALLOC_FLOOR + (1.0 - STOCK_ALLOC_FLOOR) * macro_score

    # ── 逆向修正（状态机驱动，分阶段缓调）──
    correction, reasons = compute_contrarian_correction(
        macro_score, indicator_scores, month_label,
    )

    stock_weight += correction
    stock_weight = max(STOCK_ALLOC_FLOOR, min(1.0, stock_weight))
    bond_weight = 1.0 - stock_weight

    return stock_weight, bond_weight, reasons


def _compute_style_pe(compute_type: str) -> float:
    """
    获取风格 PE 数据并计算指定指标值

    数据源：
      - stock_market_pe_lg(symbol='上证') → 大盘价值代表
      - stock_market_pe_lg(symbol='创业板') → 小盘成长代表

    Returns:
        size_pe_pct: 上证PE当前值在历史上的分位（越小越便宜）
        growth_pe_pct: 创业板PE当前值在历史上的分位（越小越便宜）
        pe_spread: 创业板PE / 上证PE 比值
    """
    pe_cache = _compute_style_pe._cache if hasattr(_compute_style_pe, "_cache") else {}
    if not pe_cache:
        try:
            df_sh = cached_api_call(ak.stock_market_pe_lg, symbol="上证", max_age_seconds=86400)
            df_gem = cached_api_call(ak.stock_market_pe_lg, symbol="创业板", max_age_seconds=86400)
            if df_sh is not None and df_gem is not None:
                sh_pe_series = df_sh["平均市盈率"].dropna().values.astype(float)
                gem_pe_series = df_gem["平均市盈率"].dropna().values.astype(float)
                pe_cache["sh_history"] = sh_pe_series
                pe_cache["gem_history"] = gem_pe_series
                pe_cache["sh_latest"] = float(sh_pe_series[-1])
                pe_cache["gem_latest"] = float(gem_pe_series[-1])
                _compute_style_pe._cache = pe_cache
        except Exception:
            _compute_style_pe._cache = {}
            return None

    sh_hist = pe_cache.get("sh_history")
    gem_hist = pe_cache.get("gem_history")
    if sh_hist is None or gem_hist is None:
        return None

    if compute_type == "size_pe_pct":
        # 上证PE在历史中的分位 → 0=历史最低（最便宜），1=历史最高（最贵）
        return float(np.searchsorted(np.sort(sh_hist), pe_cache["sh_latest"]) / len(sh_hist))

    elif compute_type == "growth_pe_pct":
        # 创业板PE在历史中的分位
        return float(np.searchsorted(np.sort(gem_hist), pe_cache["gem_latest"]) / len(gem_hist))

    elif compute_type == "pe_spread":
        # 创业板PE / 上证PE → 成长相对价值的溢价倍数
        if pe_cache["sh_latest"] > 0:
            return pe_cache["gem_latest"] / pe_cache["sh_latest"]
        return None

    return None


def get_style_indices_returns() -> dict:
    """获取风格指数近60日收益（动量信号）"""
    returns = {}
    for name, symbol in STYLE_INDICES.items():
        try:
            df = cached_api_call(
                ak.stock_zh_index_daily_tx,
                symbol=symbol,
                max_age_seconds=21600,
            )
            if df is not None and len(df) >= STYLE_LOOKBACK_DAYS:
                recent = df.tail(STYLE_LOOKBACK_DAYS)
                start_price = float(recent["close"].iloc[0])
                end_price = float(recent["close"].iloc[-1])
                returns[name] = (end_price / start_price) - 1
            else:
                returns[name] = 0.0
        except Exception:
            returns[name] = 0.0
    return returns


def determine_style_allocation(index_returns: dict = None, indicator_scores: dict = None) -> dict:
    """
    综合 PE 估值 + 动量信号判定大小盘和价值成长配置比例

    逻辑：
      - 大小盘 = 上证PE分位(50%) + 沪深300 vs 中证1000 60日动量(50%)
      - 价值成长 = 创业板PE分位(50%) + 价值 vs 成长 60日动量(50%)

    估值信号：PE分位低 = 便宜 = 超配对应风格
    动量信号：近期强 = 趋势延续 = 超配对应风格

    Returns:
        {"large": pct, "small": pct, "value": pct, "growth": pct}
    """
    index_returns = index_returns or {}
    indicator_scores = indicator_scores or {}

    # ── 大小盘决策 ──
    # PE估值信号：上证PE分位低 → 大盘便宜 → 偏向大盘
    size_pe_info = indicator_scores.get("size_pe_percentile", {})
    size_pe_pct = size_pe_info.get("raw_value", 0.5)  # PE分位
    if size_pe_pct is None or np.isnan(size_pe_pct):
        size_pe_pct = 0.5
    # PE分位 → style bias: 分位0=历史最便宜→大盘+30%, 分位1=历史最贵→大盘-30%
    size_val_signal = 0.30 * (1.0 - 2.0 * size_pe_pct)  # [-30%, +30%] 大盘超配

    # 动量信号：沪深300 - 中证1000 60日收益差
    hs300_ret = index_returns.get("hs300", 0)
    csi1000_ret = index_returns.get("csi1000", 0)
    size_mom_signal = np.clip((hs300_ret - csi1000_ret) * 3, -0.30, 0.30)  # 归一化到[-30%, +30%]

    # 综合：基准50/50 + 估值 + 动量
    size_combined = 0.50 + size_val_signal * 0.5 + size_mom_signal * 0.5
    large = np.clip(size_combined, 0.30, 0.70)
    small = 1.0 - large

    # ── 价值/成长决策 ──
    # PE估值信号：创业板PE分位低 → 成长便宜 → 偏向成长
    growth_pe_info = indicator_scores.get("growth_pe_percentile", {})
    growth_pe_pct = growth_pe_info.get("raw_value", 0.5)
    if growth_pe_pct is None or np.isnan(growth_pe_pct):
        growth_pe_pct = 0.5
    # 创业板PE分位0(便宜)→成长+30%, 分位1(贵)→价值+30%
    growth_val_signal = 0.30 * (1.0 - 2.0 * growth_pe_pct)  # [-30%, +30%] 成长超配

    # 动量信号：价值 - 成长 60日收益差
    value_ret = index_returns.get("hs300_value", 0)
    growth_ret = index_returns.get("hs300_growth", 0)
    vg_mom_signal = np.clip((value_ret - growth_ret) * 3, -0.30, 0.30)  # 正值=价值强

    # 综合：基准50/50 + 估值(偏向成长) + 动量
    vg_combined = 0.50 - growth_val_signal * 0.5 + vg_mom_signal * 0.5
    val_weight = np.clip(vg_combined, 0.30, 0.70)
    growth_weight = 1.0 - val_weight

    return {"large": round(large, 2), "small": round(small, 2),
            "value": round(val_weight, 2), "growth": round(growth_weight, 2)}


def print_macro_report(indicator_scores: dict, macro_score: float,
                       stock_w: float, bond_w: float, style_alloc: dict,
                       contrarian_reasons: list = None):
    """打印宏观模块报告"""
    contrarian_reasons = contrarian_reasons or []
    print("\n" + "=" * 60)
    print("  宏观择时报告")
    print("=" * 60)

    # 分组显示
    groups = {
        "经济增长": ["pmi", "industrial_production"],
        "通胀物价": ["cpi", "ppi", "commodity_index"],
        "流动性信用": ["bond_10y", "m2", "social_financing", "credit_spread"],
        "外部跨境": ["cn_us_spread", "fx_rate", "northbound_flow"],
        "市场情绪": ["market_volume", "margin_balance"],
        "市场热度": ["turnover_extreme", "advance_decline_ratio", "erp"],
        "风格估值": ["size_pe_percentile", "growth_pe_percentile", "pe_spread"],
        "政策信心": ["monetary_policy", "fiscal_policy", "meeting_tone", "confidence"],
    }
    for group_name, indicators in groups.items():
        print(f"\n  [{group_name}]")
        for name in indicators:
            info = indicator_scores.get(name, {})
            raw = info.get("raw_value")
            score = info.get("score", 0)
            available = info.get("available", False)
            manual = info.get("manual", False)
            proxy = info.get("proxy", False)
            stale = info.get("stale", False)
            latest_date = info.get("latest_date", "")
            stale_info = info.get("stale_info", "")
            tags = []
            if manual:
                tags.append("手动")
            if proxy:
                tags.append("代理")
            if stale:
                tags.append(f"过期-{stale_info}" if stale_info else "过期")
            elif latest_date:
                tags.append(str(latest_date))
            tag_str = f" ({','.join(tags)})" if tags else ""
            raw_str = f"{raw:.2f}" if isinstance(raw, (int, float)) else str(raw)[:10] if raw else "N/A"
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {name:<22s}: raw={raw_str:>10s}  score={score:.1f} {bar}{tag_str}")

    print(f"\n  >>> 宏观总分: {macro_score:.3f}  ({macro_score*100:.1f}%)")
    raw_w = STOCK_ALLOC_FLOOR + (1.0 - STOCK_ALLOC_FLOOR) * macro_score
    print(f"  >>> 股票仓位: {stock_w:.1%}（基准{raw_w:.1%}{' + 逆向修正' if contrarian_reasons else ''}）"
          f"  债券仓位: {bond_w:.1%}")
    if contrarian_reasons:
        for r in contrarian_reasons:
            print(f"        ⚡ {r}")
    print(f"  >>> 风格配置: 大盘{style_alloc['large']:.0%}/小盘{style_alloc['small']:.0%}  "
          f"价值{style_alloc['value']:.0%}/成长{style_alloc['growth']:.0%}")
    sh_pct = indicator_scores.get("size_pe_percentile", {}).get("raw_value")
    gem_pct = indicator_scores.get("growth_pe_percentile", {}).get("raw_value")
    spread = indicator_scores.get("pe_spread", {}).get("raw_value")
    if sh_pct is not None and gem_pct is not None:
        print(f"  >>> 估值信号: 上证PE分位{sh_pct*100:.0f}%（{'便宜' if sh_pct < 0.3 else '适中' if sh_pct < 0.7 else '偏贵'}）"
              f" | 创业板PE分位{gem_pct*100:.0f}%（{'便宜' if gem_pct < 0.3 else '适中' if gem_pct < 0.7 else '偏贵'}）"
              f" | PE比值{spread:.1f}x" if spread else "")
    print("=" * 60)
    sys.stdout.flush()
