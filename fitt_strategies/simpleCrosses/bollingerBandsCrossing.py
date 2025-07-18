from fitt_strategies.strategy_baseclass import Strategy
from fitt_strategies.utils._resistance_detection import volumeByPrice
from ta.volatility import BollingerBands
from backtesting.lib import crossover

import pandas as pd

class BBandsCrossing(Strategy):
    def init(self):
        close_series = pd.Series(self.data.Close, index=self.data.index)

        # Calculate Bollinger Bands using ta
        bb = BollingerBands(close=close_series, window=20, window_dev=2)

        # Use custom variable names as requested
        self.higher_band = self.I(bb.bollinger_hband)
        self.middle_band = self.I(bb.bollinger_mavg)

    def next(self):
        price = self.data.Close[-1]
        upper = self.higher_band[-1]
        middle = self.middle_band[-1]

        # Entry condition: breakout above upper band
        if not self.position and price > upper:
            self.buy()

        # Exit condition: price falls back below the middle band
        elif self.position and price < middle:
            self.position.close()
