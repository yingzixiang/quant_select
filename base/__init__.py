"""
共享底座（base）—— 收敛后抽出三引擎共性层

当前内容：
  - data_fetch.py   统一数据获取（日线多源回退 / 资金流向 / 指数 / 股票列表 / 流通市值）

计划（见 CONVERGENCE_DESIGN.md）：
  - cache.py        缓存层（现有 core/data_cache.py 上移）
  - factors/        因子库
  - backtest.py     统一回测引擎
  - risk.py         统一风控
  - broker/         下单层（现有 broker/ 上移）
"""
