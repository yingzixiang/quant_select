"""
下单对接层（broker）

三引擎目前只产出「信号 + 回测」，本包补齐信号 → 下单的最后一公里。

设计原则：
  - 券商无关的抽象接口 `Broker`（base.py）
  - 本地撮合的模拟盘 `PaperBroker`（paper.py），无需真实账户即可跑通全链路
  - 信号 → 下单桥 `bridge`（bridge.py），消费 short_term/output/selection_*.json
  - 真实券商适配器在 live.py 中以工厂注册方式接入，具体 SDK 待定渠道后实现
"""
from broker.base import Broker, Order, Position
from broker.paper import PaperBroker
from broker.bridge import load_selection, build_orders, akshare_price_provider

__all__ = [
    "Broker", "Order", "Position", "PaperBroker",
    "load_selection", "build_orders", "akshare_price_provider",
]
