"""
短线量化选股配置：硬约束、因子权重、模型参数
"""

# ===== 选股硬约束 =====
# 基础过滤
MIN_FLOAT_MV = 20_0000_0000    # 流通市值下限 20亿
MAX_FLOAT_MV = 500_0000_0000   # 流通市值上限 500亿
MIN_DAILY_AMOUNT = 3_0000_0000 # 日均成交额 >= 3亿
MIN_TURNOVER_RATE = 1.0        # 换手率 >= 1.0%（对齐回测最优参数）
MIN_LISTED_DAYS = 30           # 上市 >= 30天

# ===== spot 预筛参数 =====
SPOT_PRE_FILTER_TOP_N = 1000   # Spot 数据预筛 Top N（按活跃度排序进入深度分析）
SPOT_ACTIVITY_WEIGHTS = {      # 活跃度综合评分权重
    "amount": 0.50,             # 成交额权重最高
    "turnover_rate": 0.30,      # 换手率辅助
    "abs_change_pct": 0.20,     # 振幅参与（关注有波动的股票）
}

# 趋势规则
RPS_THRESHOLD = 70             # RPS(20) >= 70
MA_PERIODS = [5, 10, 20, 60]  # 均线周期

# 量价规则（对齐回测最优参数）
MIN_DAILY_CHANGE = 1.0         # 当日涨幅下限 1%
MAX_DAILY_CHANGE = 9.0         # 当日涨幅上限 9%
VOLUME_RATIO_MIN = 1.1         # 放量 >= 1.1倍5日均量
VOLUME_RATIO_MAX = 3.0         # 不放天量 < 3倍5日均量

# ===== 模型参数 =====
TRAIN_WINDOW_DAYS = 252        # 训练窗口 1年
TEST_WINDOW_DAYS = 63          # 测试窗口 3个月
LABEL_FORWARD_DAYS = 3         # 预测未来3日
LABEL_POSITIVE_THRESHOLD = 0.05  # 3日收益 > 5% 为正样本
LABEL_NEGATIVE_THRESHOLD = 0.0   # 3日收益 < 0 为负样本

# XGBoost 超参数
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
}

# 模型预测阈值
PREDICT_PROB_THRESHOLD = 0.38  # 上涨概率 >= 0.38 才入选（v2.3: 0.45→0.38，适应Sina数据源）
TOP_N_SELECT = 3               # 最终选股数量

# ===== 因子配置 =====
MAX_FACTOR_COUNT = 80          # 保留因子数量上限
MIN_FACTOR_IC = 0.03           # 因子 IC 最低阈值

# ===== 回测参数 =====
COMMISSION_RATE = 0.00025      # 手续费 万2.5
SLIPPAGE = 0.001               # 滑点 0.1%
SINGLE_STOCK_WEIGHT = 0.20     # 单票仓位 20%
MAX_POSITION_COUNT = 5         # 最多持仓 5 票
HOLD_DAYS = 3                  # 持仓周期 T+3

# 回测合格线
BACKTEST_BENCHMARKS = {
    "win_rate": 0.58,           # 胜率 >= 58%
    "profit_loss_ratio": 2.0,   # 盈亏比 >= 2.0
    "annual_return": 0.30,      # 年化 >= 30%
    "max_drawdown": 0.18,       # 最大回撤 <= 18%
    "sharpe_ratio": 1.5,        # 夏普 >= 1.5
}

# ===== 缓存配置 =====
PRICE_CACHE_TTL = 21600        # 价格数据 6小时
FLOW_CACHE_TTL = 21600         # 资金流向 6小时
INDEX_CACHE_TTL = 86400        # 指数数据 24小时

# ===== 多因子打分权重（模型辅助分） =====
FACTOR_SCORE_WEIGHTS = {
    "alpha_composite": 0.30,        # Alpha因子综合得分（↓从0.40）
    "custom_composite": 0.25,       # 自定义因子综合得分（↓从0.30）
    "money_flow_score": 0.15,       # 资金流向得分（↓从0.20）
    "technical_score": 0.15,        # 技术形态得分（↑从0.10）
    "trend_strength_score": 0.15,   # 新增：趋势强度（连续阳线+量价趋势）
}

# ===== Level 5 强势确认过滤参数 =====
STRENGTH_FILTER = {
    "enabled": True,                 # 启用强势确认过滤
    "require_bullish_candle": True,  # 当日K线需收阳
    "min_up_days_3d": 2,             # 近3日至少2日收阳（过滤一日游脉冲）
    "vol_ratio_vs_5ma": 1.3,         # 当日量 > 5日均量 × 1.3（量价共振）
    "min_excess_return": 1.0,        # 当日涨幅 > 沪深300涨幅 + 1%（非跟涨）
    "strict_mode": False,            # 严格模式: 全部条件必须满足; False: 满足3/4即可
}

MODEL_OUTPUT_DIR = "short_term/model_cache"
