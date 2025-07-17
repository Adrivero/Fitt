from backtesting import Strategy as Strategy_btModule
from abc import ABC, abstractmethod


class Strategy(Strategy_btModule,ABC):
    @abstractmethod
    def init(self):
        super().init()
        pass

    @abstractmethod
    def next(self):
        super().next()
        pass

    def __repr__(self):
        return f'<Strategy {self.__class__.__name__}>'