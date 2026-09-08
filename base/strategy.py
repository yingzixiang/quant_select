"""
统一策略接口（共享底座 base/strategy.py）

收敛重构 P3：把「选股/调仓产出」统一成 Portfolio 形状，让中线/短线两套策略插件化。
回测引擎、下单层、盯盘程序未来都只依赖 Strategy 接口，新增第 N 套策略只需实现一个类。

契约：
  - Strategy.run(**kwargs) -> Portfolio   （唯一抽象方法，薄适配器只需实现它）
  - generate_signals / build_portfolio / risk_check 为可选三步钩子，默认 no-op，
    复杂策略可 override 以复用统一编排。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


@dataclass
class Signal:
    """一条选股/调仓信号"""
    code: str
    name: str = ""
    market: str = "A"           # A / HK
    side: str = "buy"           # buy / sell
    weight: float = 0.0         # 目标仓位权重（占总资产比例，0 表示未定）
    score: float = 0.0          # 综合得分 / 上涨概率
    exit_rule: dict = field(default_factory=dict)  # 止损/止盈/卖出日等出场规则

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Portfolio:
    """策略产出的组合"""
    strategy: str = ""
    date: str = ""
    signals: list = field(default_factory=list)   # list[Signal]
    stock_weight: float = 1.0
    bond_weight: float = 0.0
    style: dict = field(default_factory=dict)     # large/small/value/growth
    meta: dict = field(default_factory=dict)      # 额外信息（macro_score 等）
    risk: dict = field(default_factory=dict)      # {"passed": bool, "violations": [], "warnings": []}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [s.to_dict() for s in self.signals]
        return d


class Strategy(ABC):
    """策略基类：所有策略（swing/monthly/未来新增）实现 run() -> Portfolio"""

    name = "base"

    @abstractmethod
    def run(self, **kwargs) -> Portfolio:
        """执行策略，产出组合"""

    # ---- 可选三步钩子（默认 no-op，供复杂策略 override）----
    def generate_signals(self, **kwargs) -> list:
        return []

    def build_portfolio(self, signals: list, **kwargs) -> Portfolio:
        return Portfolio(strategy=self.name, signals=signals)

    def risk_check(self, portfolio: Portfolio) -> dict:
        return {"passed": True, "violations": [], "warnings": []}
