from fitt_strategies.strategy_baseclass import Strategy
from fitt_strategies.utils._resistance_detection import volumeByPrice

import pandas as pd
import numpy as np

from backtesting.lib import TrailingStrategy



# Dynamic approach: Volatility adjusted lookback length. No position Sizing
class BreakoutDynamicLookbackTrailingStopLoss(TrailingStrategy):

    lookback = 20 
    ceiling = 30
    floor = 10
    initialStopLossRisk = 0.98
    trailing_stop = 0.08

    # Initial lookback
    def init(self):
        super().init()
        self.set_trailing_pct(self.trailing_stop) # Percentage at which the trailing stop will follow 


    #Every market open (Next candle)
    def next(self):
        super().next()
        price = self.data.Close[-1]
     
        # We need 31 data points
        if len(self.data.Close) < 31:
            return
        
        # Determine lookback length
        close_past31d = self.data.Close[-31:]
        today_vol = float(np.std(close_past31d[1:31]))
        yesterday_vol = float(np.std(close_past31d[0:30]))
        delta_vol = (today_vol-yesterday_vol)/today_vol # Normalized volatility


        # Increase lookback when vol increases and viceversa
        self.lookback  = round(self.lookback * (1+delta_vol))
        #Check if lookback is inside the limits
        if self.lookback>self.ceiling:
            self.lookback = self.ceiling
        elif self.lookback<self.floor:
            self.lookback = self.floor


        # Buy when close occurs above the resistance, we leave the last high as we don't want to compare yesterday's high with yesterday's close
        recent_high = self.data.High[-self.lookback:-1].max()


        if (not self.position) and (price >= recent_high):  
            self.buy()
            self.breakout_lvl  = max(self.data.High[:-1])

        # We want to set a first stoploss that prevents the strategy to be invested if price reverts at the first moment we buy:
        # Access last trade entry price, if a trade has been made
        if self.position:
            self.entry_price = self.trades[-1].entry_price
            if price<= self.initialStopLossRisk*self.entry_price:
                self.position.close()


        





