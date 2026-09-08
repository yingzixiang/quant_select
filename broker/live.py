"""
真实券商适配器注册点

在 base.Broker 契约下注册具体渠道实现。当前未绑定任何券商 SDK ——
A股实盘下单依赖券商账户 + 官方/第三方 SDK，需由使用者提供渠道后才能对接。

常见可选渠道（供选型参考）：
  - 迅投 QMT / miniQMT（xtquant，零售量化常用，需券商开通）
  - 恒生 PTrade（券商 PB 系统）
  - easytrader（同花顺/华泰客户端模拟下单，社区方案）
  - 富途 moomoo / 老虎 / 盈透 IB（港股美股，A股不通）
  - 掘金量化、聚宽、米筐（研究+模拟盘为主）

实现步骤（以新增一个渠道为例）：
  1. 在 broker/live_<channel>.py 中继承 Broker 实现 connect/place_order 等 6 个方法
  2. 在下方 BROKER_BACKENDS 注册 {名称: 工厂函数}
  3. 通过 BrokerFactory.create("channel", **kwargs) 创建实例
"""


def _paper_factory(**kwargs):
    from broker.paper import PaperBroker
    from broker.config import DEFAULT_CAPITAL
    return PaperBroker(kwargs.get("capital", DEFAULT_CAPITAL))


# 已注册后端：paper 可用；真实渠道待选型后在此补充
BROKER_BACKENDS = {
    "paper": _paper_factory,
    # "qmt": qmt_factory,        # TODO: 选定渠道后实现
    # "easytrader": easytrader_factory,
}


class BrokerFactory:
    @staticmethod
    def create(backend: str, **kwargs):
        if backend not in BROKER_BACKENDS:
            raise ValueError(f"未知券商后端 {backend!r}，可用: {list(BROKER_BACKENDS)}")
        return BROKER_BACKENDS[backend](**kwargs)
