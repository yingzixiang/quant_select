"""
intraday/config.py
A 股底仓日内 T+0 量化交易 —— 全部参数常量

按策略文档的十一层架构组织，所有可调参数集中管理。
"""

# ============================================================
# Layer 0: 标的池配置（Section 1.1）
# ============================================================
STOCK_POOL = {
    "min_avg_amount_20d": 6_0000_0000,    # 近 20 日日均成交额 >= 6 亿
    "min_turnover_rate": 3.0,              # 日均换手率 >= 3%（百分比）
    "max_turnover_rate": 16.0,             # 日均换手率 <= 16%
    "max_spread": 0.03,                    # 五档买卖价差常态 <= 0.03 元（数据不可用时跳过）
    "min_avg_amplitude_5d": 3.2,           # 近 5 日日内平均振幅 >= 3.2%
    "target_pool_size_min": 4,             # 最少监控标的数
    "target_pool_size_max": 6,             # 最多监控标的数
}

STOCK_POOL_DYNAMIC = {
    "price_limit_pct": 5.0,                # 盘中触及 ±5% 涨跌幅 → 移出
    "turnover_spike": 18.0,                # 瞬时换手率 > 18% → 移出
    "gap_extreme_pct": 2.3,                # 开盘跳空 > 2.3% → 仓位降为 0
    "consecutive_losses_eliminate": 2,     # 连续 2 笔亏损 → 当日剔除
}

# ============================================================
# Layer 1: 全域环境过滤（Section 2）
# ============================================================
ENV_FILTER = {
    "index_consecutive_bars": 3,           # 连续 K 线根数
    "index_consecutive_drop_pct": 1.1,     # 连续 3 根累计跌幅 >= 1.1%
    "advance_decline_ratio_max": 1 / 2.2,  # 涨跌比 < 1:2.2
    "index_amplitude_max": 4.5,            # 指数日内振幅 > 4.5%
    "sector_drop_threshold": -1.2,         # 板块跌幅 >= -1.2%
    "sector_falling_ratio": 0.75,          # 板块内 > 75% 个股收跌
    "trading_start": "09:35",
    "trading_end": "14:40",
    "close_only_start": "14:40",           # 此后仅平仓
    "cancel_all_time": "14:57",            # 尾盘集合竞价前撤单
}

# ============================================================
# Layer 2: 日内行情分型（Section 3）
# ============================================================
MARKET_CLASSIFIER = {
    "center_window_start": "09:35",
    "center_window_end": "10:05",          # 30 分钟窗口
    "center_window_bars_1min": 30,
    "center_window_bars_5min": 6,
    "recheck_interval_minutes": 10,        # 每 10 分钟复核
    # 滞回区间（Hysteresis）
    "hysteresis": {
        "range_to_trend_price_pct": 0.9,   # 偏离 > 0.9% → 趋势
        "range_to_trend_bar_count": 3,     # 持续 >= 3 根 K 线
        "trend_to_range_price_pct": 0.6,   # 偏离 < 0.6% → 震荡
        "trend_to_range_bar_count": 5,     # 持续 >= 5 根 K 线
        "confirm_required": 2,             # 连续 2 次复核确认 → 切换
    },
    "reversal_volume_ratio": 1.8,          # 量比 >= 1.8 确认变盘
    # MA20 方向快速通道（补充价格偏离度检测不到的趋势）
    "ma20_slope_fast_path": {
        "enabled": True,                   # 启用 MA20 斜率快速通道
        "slope_bars": 15,                  # 回看 K 线数
        "slope_threshold_pct": 0.25,       # MA20 斜率 > 0.25%/bar → 强制多头趋势
        "consecutive_bars": 8,             # 连续 N 根 bar MA20 同向 → 触发
    },
}

# ============================================================
# Layer 3: 自适应指标体系（Section 4）
# ============================================================
INDICATORS = {
    # ATR
    "atr_window": 20,                      # ATR 计算窗口（回测 15-30 取最优 20）
    "atr_multiplier_default": 1.5,         # 默认 ATR 系数
    "atr_buffer": 0.6,                     # 缓冲区间 ±0.6 ATR（v3.4: 放宽入口+精选过滤）
    # 均线
    "ma_short": 5,                         # MA5 - 仅用于动态止损跟踪
    "ma_core": 10,                         # MA10 - 核心趋势线（入场信号）
    "ma_direction": 20,                    # MA20 - 日内大方向
    # 量能
    "volume_ratio_threshold_default": 1.1, # 默认相对量比阈值
    "volume_ratio_lookback_days": 5,       # 历史同时段均量回看天数
    "breakout_volume_ratio": 1.8,          # 变盘突破确认量比
    "reversal_volume_check_bars": 3,       # 拐点量能与前 N 分钟均值对比
    # K 线形态
    "candle_lower_shadow_ratio": 2.0,      # 下影 > 实体 2 倍 → 禁低吸
    "candle_upper_shadow_ratio": 2.0,      # 上影 > 实体 2 倍 → 禁高抛
    "momentum_exhaustion_bars": 3,         # 连续 N 根实体缩小 → 动能衰竭
    # 入口精选过滤（v3.4: 放宽缓冲区 + K线方向过滤 → 提升胜率）
    "require_candle_direction": True,      # 做多需收阳、做空需收阴
    "require_volume_increasing": False,    # 不做量能递增过滤（过严，杀太多信号）
    "min_relative_volume_long": 1.1,       # 做多最小量比（保持原值）
    # 标的池最低 ATR 过滤（v3.4: 剔除低波动标的）
    "min_atr_5min": 0.08,                  # 5min ATR 不低于 0.08元（约0.5%波动率）
}

# ============================================================
# Layer 4: 分场景交易规则（Section 5）
# ============================================================
SCENARIO_RULES = {
    "range_trading": {
        "staggered_tp_1_ratio": 0.60,      # 阶梯止盈①: 卖出 60%
        "staggered_tp_2_pct": 1.0,         # 阶梯止盈②: 剩余盈利达 1.0%
        "hard_stop_atr": 0.5,              # 跌破下轨 - 0.5 ATR
        "hard_stop_pct": 0.7,              # 单笔亏损达 0.7%
        "reverse_retreat_atr": 0.5,        # 反向运行超 0.5 ATR → 撤离（v3.2: 0.3→0.5 减少误撤离）
        "breakout_cover_atr": 1.0,         # 突破上轨 + 1 ATR → 踏空止损
    },
    "bull_trend": {
        "entry_zone": "MA10_VWAP_support", # MA10 + 分时均价双重支撑
        "trailing_stop_ma": "MA5",         # 跌破 MA5 立即卖出
        "stop_loss_pct": 0.6,              # 跌破均价 0.6%
    },
    "bear_trend": {
        "entry_zone": "MA10_VWAP_resistance", # MA10 + 分时均价双重压力
        "cover_drop_pct": 0.9,             # 回落 0.9% 接回
        "stop_loss_bars": 20,              # 突破近 20 根 K 线最高价 → 止损
    },
    "reversal": {
        "short_term_bars": 10,             # 短期高低点参考 K 线数
        "pressure_atr": 0.3,               # 短期压力 = 前 N 根最高 + 0.3 ATR
        "cover_zone_atr": 0.2,             # 接回区 = 下轨 ± 0.2 ATR
    },
    "gap": {
        "gap_ignore_pct": 0.5,             # < 0.5% → 忽略
        "gap_small_start": 0.5,            # 0.5%~1.5% → 20% 仓位
        "gap_small_end": 1.5,
        "gap_large_start": 1.5,            # 1.5%~2.3% → 10% 仓位
        "gap_large_end": 2.3,
        "gap_extreme_pct": 2.3,            # > 2.3% → 0% 仓位
        "tp_gap_close_pct": 0.15,          # 缺口回补至距开盘价 ±0.15% 平仓
        "sl_gap_reverse_pct": 0.5,         # 反向突破开盘价 0.5% 止损
    },
}

# ============================================================
# Layer 5: 动态仓位管理（Section 6）
# ============================================================
POSITION = {
    "range_normal": 0.30,                  # 常规震荡市: 底仓的 30%
    "trend_confident": 0.45,               # 趋势明确: 底仓的 45%
    "weak_choppy": 0.15,                   # 大盘偏弱: 底仓的 15%
    "gap_medium": 0.20,                    # 跳空 0.5%-1.5%: 底仓的 20%
    "gap_large": 0.10,                     # 跳空 1.5%-2.3%: 底仓的 10%
    "gap_extreme": 0.0,                    # 跳空 > 2.3%: 0%
    "single_stock_loss_halt_pct": -1.0,    # 单标的累计亏损 >= -1.0% → 停止
    "account_loss_reduced_pct": 0.05,      # 触发全账户止损后降为 5%
    "single_stock_max_total": 0.08,        # 单票做 T 总仓位 <= 总资产 8%
    "account_max_total": 0.20,             # 全账户做 T 总仓位 <= 总资产 20%
    "max_rounds_per_stock": 3,             # 单日最多 3 轮完整回转
}

# ============================================================
# Layer 6.2: 动态全账户止损（Section 6.2）
# ============================================================
ACCOUNT_STOP_LOSS = {
    "profit_floor_high": 1.0,              # 累计盈利 >= 1.0% → 放宽阈值
    "profit_floor_neutral": -0.5,          # 盈亏在 -0.5%~+1.0% → 标准阈值
    "loss_floor_low": -1.5,                # 累计亏损已达 -1.5% → 收紧
    "threshold_profit": 2.5,               # 有利润垫: 2.5%
    "threshold_neutral": 1.8,              # 标准: 1.8%
    "threshold_loss": 1.5,                 # 加速收紧: 1.5%
}

# ============================================================
# Layer 6: 全域硬风控（Section 7）
# ============================================================
RISK = {
    "single_trade_loss_max_pct": -0.7,     # 单笔最大亏损 -0.7%
    "single_stock_daily_loss_max_pct": -1.4,  # 单标的单日累计最大亏损 -1.4%
    "order_timeout_seconds": 60,           # 委托超时 1 分钟撤单
    "slippage_offset": 0.02,              # 委托价格偏移 ±0.02 元
    "signal_pause_minutes": 15,            # 连续 2 次未成交 → 暂停 15 分钟
    "signal_pause_threshold": 2,           # 连续 N 次未成交触发暂停
    "price_limit_up_lock": True,           # 涨停封死停止买入
    "price_limit_down_lock": True,         # 跌停封死禁止卖出
}

# ============================================================
# Layer 7: 分行情参数矩阵（Section 8）
# ============================================================
PARAMETER_MATRICES = {
    "narrow_range": {
        "name": "窄幅震荡（弱行情）",
        "atr_multiplier": 1.3,
        "take_profit_pct": 0.9,
        "stop_loss_pct": 0.6,
        "volume_ratio_threshold": 1.05,
        "base_position_ratio": 0.25,
    },
    "normal": {
        "name": "常态波动（中性行情）",
        "atr_multiplier": 1.5,
        "take_profit_pct": 1.0,
        "stop_loss_pct": 0.7,
        "volume_ratio_threshold": 1.10,
        "base_position_ratio": 0.30,
    },
    "high_active": {
        "name": "高活跃行情（热点/牛市）",
        "atr_multiplier": 1.7,
        "take_profit_pct": 1.2,
        "stop_loss_pct": 0.8,
        "volume_ratio_threshold": 1.15,
        "base_position_ratio": 0.40,
    },
}

# ============================================================
# Layer 9: 回测验收标准（Section 10）
# ============================================================
BACKTEST_ACCEPTANCE = {
    "min_win_rate": 0.57,
    "min_profit_loss_ratio": 1.3,
    "max_daily_drawdown": 0.07,            # 最大单日回撤 < 7%
    "max_false_signal_rate": 0.18,         # 无效假信号占比 < 18%
}

# 交易成本
TRANSACTION_COST = {
    "commission_rate": 0.00025,            # 手续费 万 2.5
    "slippage_rate": 0.0005,               # 滑点 0.05% 每边（往返 0.08%）
    "stamp_tax_rate": 0.0,                 # A 股卖出印花税（底仓 T+0 按现有规则）
}

# ============================================================
# Layer 10: 参数敏感性搜索网格（Section 10.2）
# ============================================================
PARAMETER_SEARCH_GRID = {
    "atr_multiplier":      (1.2, 1.8, 0.1),     # (start, end, step)
    "atr_window":          (10,  30,  5),
    "volume_ratio_threshold": (1.05, 1.30, 0.05),
    "stop_loss_pct":       (0.5, 1.0, 0.1),
    "staggered_tp_ratio":  (0.50, 0.80, 0.10),
    "hysteresis_width":    (0.1, 0.5, 0.1),
}

# ============================================================
# 数据缓存配置
# ============================================================
DATA_CACHE_TTL = {
    "minute_kline": 21600,     # 6 小时
    "index_minute": 21600,
    "sector_minute": 21600,
    "sector_spot": 86400,      # 24 小时
    "stock_list": 86400,
    "sector_map": 86400,
}

# ============================================================
# 回测默认参数
# ============================================================
BACKTEST_DEFAULTS = {
    "initial_capital": 1_000_000,           # 初始资金 100 万
    "period": "5",                          # 默认 5 分钟 K 线
    "default_param_set": "normal",          # 默认参数矩阵
    "base_position_shares": 10000,          # 默认底仓股数（无真实数据时的假设）
}

# ============================================================
# 数据缺失降级开关
# ============================================================
DATA_DEGRADATION = {
    "order_book_available": False,          # 五档盘口 - AKShare 不支持
    "bs_ratio_available": False,            # BS 比率 - 5 分钟 bar 不含
    "sector_minute_available": True,        # 板块分钟 - 尝试但接受失败
    "advance_decline_available": False,     # 涨跌家数 - 无直接 API
    "tick_level_data_available": False,     # 逐笔成交 - 仅当日实时
}
