"""
18项宏观指标：API映射 + 打分阈值表

每项指标结构:
  - api_func: akshare函数名(str) 或 None(需手动输入)
  - api_kwargs: API额外参数
  - value_key: 从返回DataFrame中提取值的列名
  - value_index: 取第几行（默认0=最新）
  - scoring_type: "threshold"（分段阈值）或 "percentile"（历史分位）
  - direction: "positive"（越高越好）或 "negative"（越低越好）或 "center"（区间最优）
  - thresholds: 分段阈值 {breakpoint: score}
  - manual_default: 手动输入默认值(2-10分)
"""

MACRO_INDICATORS = {
    # ===== 2.2.1 经济增长组（8%）=====
    "pmi": {
        "api_func": "macro_china_pmi",
        "api_kwargs": {},
        "value_key": "制造业-指数",
        "date_key": "月份",
        "date_format": "cn_month",
        "newest_first": True,
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {52: 10, 50: 8, 48: 5, 46: 3, 0: 0},
    },
    "industrial_production": {
        "api_func": "macro_china_gyzjz",
        "api_kwargs": {},
        "value_key": "同比增长",
        "date_key": "月份",
        "date_format": "cn_month",
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {8: 10, 6: 8, 4: 6, 2: 4, 0: 2, float("-inf"): 0},
    },

    # ===== 2.2.2 通胀物价组（4.5%）=====
    "cpi": {
        "api_func": "macro_china_cpi",
        "api_kwargs": {},
        "value_key": "全国-同比增长",
        "date_key": "月份",
        "date_format": "cn_month",
        "newest_first": True,
        "scoring_type": "center",
        "direction": "center",
        # (lower_bound, upper_bound) → score
        "center_ranges": [
            ((2, 3), 10),
            ((1, 2), 8), ((3, 4), 8),
            ((0, 1), 6), ((4, 5), 6),
            ((-1, 0), 4), ((5, 6), 4),
            ((float("-inf"), -1), 2), ((6, float("inf")), 2),
        ],
    },
    "ppi": {
        "api_func": "macro_china_ppi",
        "api_kwargs": {},
        "value_key": "当月同比增长",
        "date_key": "月份",
        "date_format": "cn_month",
        "newest_first": True,
        "scoring_type": "center",
        "direction": "center",
        "center_ranges": [
            ((3, 5), 10),
            ((1, 3), 8), ((5, 7), 8),
            ((0, 1), 6), ((7, 9), 6),
            ((-3, 0), 4), ((9, 11), 4),
            ((float("-inf"), -3), 2), ((11, float("inf")), 2),
        ],
    },
    "commodity_index": {
        "api_func": "macro_china_commodity_price_index",
        "api_kwargs": {},
        "value_key": "近1年涨跌幅",
        "scoring_type": "center",
        "direction": "center",
        "center_ranges": [
            ((0, 10), 10),
            ((-5, 0), 8), ((10, 20), 8),
            ((-10, -5), 6), ((20, 30), 6),
            ((-20, -10), 4), ((30, 40), 4),
            ((float("-inf"), -20), 2), ((40, float("inf")), 2),
        ],
    },

    # ===== 2.2.3 流动性与信用组（10%）=====
    "bond_10y": {
        "api_func": "bond_zh_us_rate",
        "api_kwargs": {},
        "value_key": "中国国债收益率10年",
        "scoring_type": "threshold",
        "direction": "negative",
        "thresholds": {2.5: 10, 2.7: 8, 2.9: 6, 3.1: 4, float("inf"): 2},
    },
    "m2": {
        "api_func": "macro_china_money_supply",
        "api_kwargs": {},
        "value_key": "货币和准货币(M2)-同比增长",
        "date_key": "月份",
        "date_format": "cn_month",
        "newest_first": True,
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {12: 10, 11: 8, 10: 6, 9: 4, 0: 2},
    },
    "social_financing": {
        "api_func": "macro_china_shrzgm",
        "api_kwargs": {},
        "value_key": "社会融资规模增量",
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {20: 10, 10: 8, 0: 6, -10: 4, float("-inf"): 2},
        "compute_yoy": True,  # 需要计算同比
    },
    "credit_spread": {
        "api_func": None,  # bond_china_yield AA数据过旧(2021)，需手动输入
        "manual_default": 6,
        "scoring_type": "threshold",
        "direction": "negative",
        "thresholds": {0.8: 10, 1.0: 8, 1.2: 6, 1.5: 4, float("inf"): 2},
    },

    # ===== 2.2.4 外部跨境组（4.5%）=====
    "cn_us_spread": {
        "api_func": "bond_zh_us_rate",
        "api_kwargs": {},
        "value_key": None,  # 计算：中国10Y - 美国10Y
        "compute": "cn_minus_us",  # 特殊标记，需手动计算
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {0: 10, -0.5: 8, -1.0: 6, -1.5: 4, float("-inf"): 2},
    },
    "fx_rate": {
        "api_func": "fx_spot_quote",
        "api_kwargs": {},
        "value_key": "买报价",
        "scoring_type": "threshold",
        "direction": "negative",
        "thresholds": {6.7: 10, 6.9: 8, 7.1: 6, 7.3: 4, float("inf"): 2},
        "fx_filter": "USD/CNY",  # 过滤特定货币对
    },
    "northbound_flow": {
        "api_func": "stock_hsgt_hist_em",
        "api_kwargs": {},
        "value_key": "当日成交净买额",
        "scoring_type": "percentile",
        "direction": "positive",
        "percentile_map": {0.8: 10, 0.6: 8, 0.4: 6, 0.2: 4, 0.0: 2},
        "aggregate": "monthly_sum",  # 月合计
        "lookback": 252,  # 近1年交易日
    },

    # ===== 2.2.5 市场情绪组（3%）=====
    "market_volume": {
        "api_func": "stock_zh_a_spot",
        "api_kwargs": {},
        "value_key": "成交额",
        "scoring_type": "percentile",
        "direction": "positive",
        "percentile_map": {0.8: 10, 0.6: 8, 0.4: 6, 0.2: 4, 0.0: 2},
        "aggregate": "sum",  # 全市场合计
    },
    "margin_balance": {
        "api_func": "macro_china_market_margin_sh",
        "api_kwargs": {},
        "value_key": "融资融券余额",
        "scoring_type": "threshold",
        "direction": "positive",
        "thresholds": {10: 10, 5: 8, 0: 6, -5: 4, float("-inf"): 2},
        "compute_mom": True,  # 计算环比增速
        "dual_source": "macro_china_market_margin_sz",  # 需要加总沪深两市
    },

    # ===== 2.2.6 政策与信心组（6%）=====
    "monetary_policy": {
        "api_func": None,  # 定性指标，需手动输入
        "manual_default": 6,
        "scoring_type": "manual",
    },
    "fiscal_policy": {
        "api_func": None,  # 定性指标，需手动输入
        "manual_default": 6,
        "scoring_type": "manual",
    },
    "meeting_tone": {
        "api_func": None,  # 定性指标，需手动输入
        "manual_default": 6,
        "scoring_type": "manual",
    },
    "confidence": {
        "api_func": None,  # macro_china_consumer_confidence 不存在
        "manual_default": None,
        "scoring_type": "proxy",
        "proxy_source": ["pmi", "industrial_production"],  # 取两者得分均值
    },

    # ===== 2.2.7 市场热度/冷度组（8%）=====
    "turnover_extreme": {
        "api_func": "stock_zh_index_daily_tx",
        "api_kwargs": {"symbol": "sh000001"},
        "value_key": "amount",
        "scoring_type": "center",
        "direction": "center",
        "compute": "turnover_ratio",  # 今日成交额 / 20日均成交额
        "center_ranges": [
            ((0.7, 1.3), 10),           # 正常量
            ((0.5, 0.7), 8), ((1.3, 1.8), 8),   # 偏缩/偏放
            ((0.3, 0.5), 6), ((1.8, 2.5), 6),   # 萎缩/放量
            ((0.1, 0.3), 4), ((2.5, 3.5), 4),   # 地量/天量
            ((float("-inf"), 0.1), 2), ((3.5, float("inf")), 2),  # 极端
        ],
    },
    "advance_decline_ratio": {
        "api_func": "stock_zh_index_daily_tx",
        "api_kwargs": {"symbol": "sh000001"},
        "value_key": "close",
        "scoring_type": "center",
        "direction": "center",
        "compute": "advance_decline_20d",  # 近20日收涨天数占比
        "center_ranges": [
            ((0.45, 0.55), 10),           # 涨跌互现（健康轮动）
            ((0.35, 0.45), 8), ((0.55, 0.65), 8),   # 偏弱/偏强
            ((0.25, 0.35), 6), ((0.65, 0.75), 6),   # 弱势/强势
            ((0.15, 0.25), 4), ((0.75, 0.85), 4),   # 恐慌/亢奋
            ((float("-inf"), 0.15), 2), ((0.85, float("inf")), 2),  # 极端
        ],
    },
    "erp": {
        "api_func": "stock_market_pe_lg",
        "api_kwargs": {"symbol": "上证"},
        "value_key": "平均市盈率",
        "scoring_type": "threshold",
        "direction": "positive",
        "compute": "erp",  # (100/PE) - 10Y国债收益率
        "thresholds": {6: 10, 4: 8, 2: 6, 1: 4, 0: 2, float("-inf"): 1},
    },

    # ===== 2.2.8 风格估值组（5%）=====
    # 大小盘估值：上证PE历史分位 → PE越低大盘越便宜 → 得分越高
    "size_pe_percentile": {
        "api_func": None,  # 独立函数 _compute_style_pe 处理
        "manual_default": None,
        "scoring_type": "threshold",
        "direction": "negative",
        "compute": "size_pe_pct",
        "thresholds": {0.15: 10, 0.30: 8, 0.50: 6, 0.70: 4, float("inf"): 2},
    },
    # 成长估值：创业板PE历史分位 → PE越低成长越便宜 → 得分越高
    "growth_pe_percentile": {
        "api_func": None,
        "manual_default": None,
        "scoring_type": "threshold",
        "direction": "negative",
        "compute": "growth_pe_pct",
        "thresholds": {0.15: 10, 0.30: 8, 0.50: 6, 0.70: 4, float("inf"): 2},
    },
    # 风格价差：创业板PE / 上证PE 比值 → 极端偏离=泡沫/低估信号
    "pe_spread": {
        "api_func": None,
        "manual_default": None,
        "scoring_type": "center",
        "direction": "center",
        "compute": "pe_spread",
        "center_ranges": [
            ((2.5, 3.5), 10),           # 正常区间
            ((2.0, 2.5), 8), ((3.5, 4.0), 8),   # 偏价值/偏成长
            ((1.5, 2.0), 6), ((4.0, 5.0), 6),   # 价值极端/成长泡沫
            ((0.0, 1.5), 4), ((5.0, 6.0), 4),   # 恐慌/狂热
            ((float("-inf"), 0.0), 2), ((6.0, float("inf")), 2),
        ],
    },
}

# 行业 → 板块 映射（小行业统计量不足时回退）
INDUSTRY_TO_SECTOR = {
    "银行": "金融", "非银金融": "金融",
    "钢铁": "周期", "有色金属": "周期", "基础化工": "周期",
    "石油石化": "周期", "煤炭": "周期", "建筑材料": "周期",
    "机械设备": "制造", "汽车": "制造", "电力设备": "制造",
    "国防军工": "制造", "建筑装饰": "制造",
    "食品饮料": "消费", "家用电器": "消费", "纺织服饰": "消费",
    "轻工制造": "消费", "商贸零售": "消费", "社会服务": "消费",
    "农林牧渔": "消费", "美容护理": "消费",
    "电子": "TMT", "计算机": "TMT", "通信": "TMT", "传媒": "TMT",
    "医药生物": "医药",
    "公用事业": "公用", "交通运输": "公用", "环保": "公用",
    "房地产": "地产",
    "综合": "其他",
}
