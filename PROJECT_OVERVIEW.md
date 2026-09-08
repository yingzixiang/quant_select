# 量化选股模型 - 项目结构图谱

## 整体架构

```
main.py                          # 程序入口，串联全流程（A股+港股）
├── config/settings.py           # 配置中心（日期、因子权重、风控参数）
├── utils/common.py              # 工具层（数据清洗、日期格式化）
├── core/stock_pool.py           # 股票池（A股全市场 + 港股Top200成交额 + 风控过滤）
├── core/factor_calc.py          # 因子层（A股批量因子 / 港股逐只因子 + Z-score标准化）
├── core/selector.py             # 选股层（加权打分 + Top N 筛选）
├── core/trader.py               # 回测层（A股持仓模拟 + 收益率计算）
└── risk/filter_rule.py          # 风控规则（预留扩展）
```

## 执行流程

```
[1. 获取股票池] ──→ [2. 风控过滤] ──→ [3. 计算因子] ──→ [4. 市场内标准化] ──→ [5. 加权打分选股] ──→ [6. 组合回测]
       │                                      │
       ├── A股: stock_info_a_code_name()      ├── A股: stock_yjbb_em() 批量 + 逐只PB/价格
       └── 港股: stock_hk_spot() Top200       └── 港股: stock_hk_financial_indicator_em() + stock_hk_daily()
```

| 步骤 | 模块 | 数据源 API | 输出 |
|------|------|------|------|
| (1) 获取A股池 | `stock_pool.get_total_stocks()` | `ak.stock_info_a_code_name()` | 全量 A 股 code + name |
| (1) 获取港股池 | `stock_pool.get_hk_stocks()` | `ak.stock_hk_spot()` | Top 200 港股（按成交额） |
| (2) 风控过滤 | `stock_pool.filter_risk_stocks()` | — | 剔除 ST 后的代码列表 |
| (3) A股因子 | `factor_calc.get_stock_factors()` | PB/财务/日线 API | A股因子 DataFrame |
| (3) 港股因子 | `factor_calc.get_hk_stock_factors()` | 港股财务/日线 API | 港股因子 DataFrame |
| (4) 市场内标准化 | `factor_calc.factor_standardize()` | — | 各市场独立Z-score |
| (5) 打分选股 | `selector.select_target_stocks()` | — | Top 25 标的代码（跨市场） |
| (6) 组合回测 | `trader.simulate_holding_return()` | `ak.stock_zh_a_hist()` | A股组合净值 + 收益率 |

## 选股因子

| 因子 | 方向 | 权重 | A股数据来源 | 港股数据来源 |
|------|------|------|----------|----------|
| PB（市净率） | 负向（低估值） | -0.35 | `ak.stock_zh_valuation_baidu(symbol, "市净率")` | `ak.stock_hk_financial_indicator_em(symbol)` → 市净率 |
| ROE（净资产收益率） | 正向（高盈利） | +0.30 | `ak.stock_yjbb_em(date)` 批量 | `ak.stock_hk_financial_indicator_em(symbol)` → 股东权益回报率(%) |
| profit_growth（利润增速） | 正向（高成长） | +0.20 | `ak.stock_yjbb_em(date)` 批量 → 净利润-同比增长 | `ak.stock_hk_financial_indicator_em(symbol)` → 净利润滚动环比增长(%) |
| mom_60（60日动量） | 正向（趋势） | +0.15 | `ak.stock_zh_a_hist(symbol)` 收盘价 pct_change(59) | `ak.stock_hk_daily(symbol)` close pct_change(59) |

**打分逻辑**：各市场内部因子 Z-score 标准化 → 合并 → 加权求和 → 取 Top 25

> 注意：A股利润增速为 YoY（同比增长），港股为 QoQ（环比增长），两者指标口径不同，因此在各自市场内部独立标准化后再合并打分。

## 风控规则

- 剔除 ST 股票（名称含 "ST"）
- ROE 低于 -10 或高于 50 的异常值截断
- 市净率 ≤ 0 的异常值过滤
- 港股：按日成交额排序取 Top 200，排除流动性不足的标的

## 回测参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SELECT_STOCK_NUM | 25 | 每期选股数量 |
| HOLD_DAYS | 30 | 持仓天数 |
| MIN_LISTED_DAYS | 365 | 最小上市天数 |
| SINGLE_STOCK_MAX_WEIGHT | 0.06 | 单只最大仓位 |

## 数据源

- **AKShare**：完全免费，无需注册，无数据时间限制，实时更新
  - A股基础信息：`ak.stock_info_a_code_name()`
  - A股日线行情：`ak.stock_zh_a_hist(symbol)`
  - A股估值数据（PB）：`ak.stock_zh_valuation_baidu(symbol, "市净率")`
  - A股批量财务（ROE + 利润增速）：`ak.stock_yjbb_em(date="20260331")`
  - 港股行情列表：`ak.stock_hk_spot()`
  - 港股日线行情：`ak.stock_hk_daily(symbol)`
  - 港股财务指标（PB/ROE/增速）：`ak.stock_hk_financial_indicator_em(symbol)`

## 运行方式

```bash
# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 启动选股流程（无需配置账号）
python3 main.py
```

## 输出产物

| 产物 | 路径 |
|------|------|
| 精选标的列表（A股+港股） | 控制台输出（含代码 + 名称 + 市场标识） |
| A股组合净值 + 收益率 | 控制台输出 |
