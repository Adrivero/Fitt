import datetime  
import os.path  
import sys  

from backtesting.lib import crossover
from backtesting.test import SMA, GOOG
from backtesting.lib import Backtest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fitt_strategies.strategy_baseclass import Strategy

# Simple moving average strategy with a 20 and 40 day MA 
class SmaCross(Strategy):
    def init (self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 20)
        self.ma2 = self.I(SMA, price, 40)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

