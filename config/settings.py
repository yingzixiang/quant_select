# ===== 回测日期 =====
REBALANCE_MONTH = "2026-07"
START_DATE = "20260501"
END_DATE = "20260701"
TEST_END_DATE = "20260720"

# ===== 基准标的 =====
BENCHMARK = "000300"

# ===== 选股与持仓 =====
SELECT_STOCK_NUM = 30          # 每期选股数量
A_SHARE_MIN_COUNT = 20         # A股最少数量
HK_MAX_COUNT = 10              # 港股最多数量
SINGLE_STOCK_MAX_WEIGHT = 0.05 # 单只个股上限
INDUSTRY_MAX_WEIGHT = 0.20     # 单一行业上限

# ===== 宏观择时 =====
MACRO_TOTAL_WEIGHT = 0.30      # 宏观模块总权重
MICRO_TOTAL_WEIGHT = 0.70      # 微观模块总权重
STOCK_ALLOC_FLOOR = 0.15       # 股票仓位下限（历史回测校准）
STOCK_ALLOC_CEILING = 1.00     # 股票仓位上限

# 宏观指标权重（合计 = 0.30）
MACRO_INDICATOR_WEIGHTS = {
    "pmi": 0.04, "industrial_production": 0.04,
    "cpi": 0.015, "ppi": 0.015, "commodity_index": 0.015,
    "bond_10y": 0.03, "m2": 0.03, "social_financing": 0.02, "credit_spread": 0.02,
    "cn_us_spread": 0.015, "fx_rate": 0.015, "northbound_flow": 0.015,
    "market_volume": 0.02, "margin_balance": 0.01,
    "monetary_policy": 0.015, "fiscal_policy": 0.015, "meeting_tone": 0.015,
    "confidence": 0.015,
    # 市场热度/冷度
    "turnover_extreme": 0.025, "advance_decline_ratio": 0.025, "erp": 0.03,
    # 风格估值（大小盘/价值成长 PE 分位）
    "size_pe_percentile": 0.015, "growth_pe_percentile": 0.015, "pe_spread": 0.02,
}

# 手动输入宏观指标默认值（2-10分制，默认中性6分）
MANUAL_MACRO_DEFAULTS = {
    "credit_spread": 6,   # 信用利差
    "monetary_policy": 6, # 货币政策
    "fiscal_policy": 6,   # 财政政策
    "meeting_tone": 6,    # 会议定调
}

# ===== 微观因子权重（12项，合计 = 0.70 归一化到 1.0） =====
MICRO_FACTOR_WEIGHT = {
    # 估值因子（15%）
    "pb":             -0.05,
    "pe_ttm":         -0.05,
    "dividend_yield":  0.05,
    # 质量因子（15%）
    "roe":             0.05,
    "debt_ratio":     -0.05,
    "gross_margin_yoy": 0.05,
    # 成长因子（15%）
    "profit_growth":   0.05,
    "revenue_growth":  0.05,
    "rd_intensity_growth": 0.05,
    # 量价因子（10%）
    "mom_60":          0.05,
    "volatility_30":  -0.05,
    # 流动性因子（5%）
    "avg_amount_20":   0.05,
}

# 因子方向映射（用于Z-Score方向统一）
MICRO_FACTOR_DIRECTION = {k: "positive" if v > 0 else "negative"
                          for k, v in MICRO_FACTOR_WEIGHT.items()}

# ===== 风格轮动 =====
STYLE_LOOKBACK_DAYS = 60
STYLE_THRESHOLD = 0.02  # ±2%判定阈值

STYLE_INDICES = {
    "hs300": "sh000300",
    "csi1000": "sh000852",
    "hs300_value": "sh000919",
    "hs300_growth": "sh000921",
}

# ===== 风控参数 =====
ROE_MIN = -10
ROE_MAX = 50
MIN_LISTED_DAYS = 365
MIN_DAILY_AMOUNT = 50_000_000    # 近20日日均成交额下限
MAX_DEBT_RATIO = 0.80            # 资产负债率上限

# 后置风控阈值
POST_RISK = {
    "single_stock_max_weight": 0.05,
    "single_industry_max_weight": 0.20,
    "value_pe_pb_ratio_max": 1.5,   # vs 沪深300
    "growth_pe_pb_ratio_max": 2.0,  # vs 沪深300
    "volatility_ratio_max": 1.2,    # vs 沪深300
}

# ===== 缓存配置 =====
CACHE_DIR = "cache"
MACRO_CACHE_TTL = 86400      # 宏观数据 24小时
FINANCIAL_CACHE_TTL = 86400  # 财务数据 24小时
PRICE_CACHE_TTL = 21600      # 价格数据 6小时

# ===== 兼容旧配置 =====
FACTOR_WEIGHT = {
    "pb": -0.35,
    "roe": 0.30,
    "profit_growth": 0.20,
    "mom_60": 0.15,
}
HOLD_DAYS = 30
