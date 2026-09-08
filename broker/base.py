"""
券商抽象接口：定义下单/撤单/查询的标准契约

真实券商适配器只需继承 Broker 并实现以下方法，其余链路（信号桥、模拟盘）无需改动。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class Order:
    """一笔委托"""
    code: str
    side: str = "buy"           # buy / sell
    quantity: int = 0           # 股数
    price: float = 0.0          # 委托价
    name: str = ""
    order_type: str = "limit"   # limit / market
    status: str = "pending"     # pending / filled / cancelled / rejected
    order_id: str = ""
    filled_price: float = 0.0
    reason: str = ""            # 下单原因（信号/止损/止盈/人工）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    filled_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Order":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class Position:
    """持仓"""
    code: str
    quantity: int = 0
    avg_price: float = 0.0
    name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker(ABC):
    """券商接口：所有实现（模拟盘/真实券商）需满足此契约"""

    @abstractmethod
    def connect(self) -> bool:
        """建立连接/登录，成功返回 True"""

    @abstractmethod
    def get_cash(self) -> float:
        """查询可用资金"""

    @abstractmethod
    def get_positions(self) -> list:
        """查询当前持仓，返回 list[Position]"""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """下单，返回带 order_id / status / filled_price 的订单"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""

    @abstractmethod
    def get_orders(self) -> list:
        """查询当日/历史委托，返回 list[Order]"""
