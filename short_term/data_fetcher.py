"""
兼容 shim：数据获取实现已上移到共享底座 base/data_fetch.py（收敛重构 P1）

保留本文件是为了让现有 `from short_term.data_fetcher import ...` 调用方无需改动。
新增代码请直接 `from base.data_fetch import ...`。
"""
from base.data_fetch import (
    safe_float,
    fetch_a_stock_list,
    fetch_daily_kline_batch,
    fetch_money_flow_batch,
    fetch_index_daily,
    fetch_market_cap_float,
)

__all__ = [
    "safe_float",
    "fetch_a_stock_list",
    "fetch_daily_kline_batch",
    "fetch_money_flow_batch",
    "fetch_index_daily",
    "fetch_market_cap_float",
]
