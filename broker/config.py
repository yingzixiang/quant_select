"""
下单对接配置：交易成本、仓位约束、模拟盘状态路径
"""
# ===== 交易成本 =====
COMMISSION_RATE = 0.00025   # 手续费 万2.5（买卖各收）
STAMP_TAX_RATE = 0.0005     # 印花税 0.05%（仅卖出，2023-08 后标准）

# ===== A股交易规则 =====
MIN_LOT = 100               # 最小整手股数

# ===== 仓位约束（与短线系统对齐）=====
MAX_POSITION_COUNT = 5      # 最多同时持仓票数
SINGLE_STOCK_WEIGHT = 0.20  # 单票仓位 20%

# ===== 模拟盘 =====
DEFAULT_CAPITAL = 1_000_000
PAPER_STATE_FILE = "output/paper_account.json"
