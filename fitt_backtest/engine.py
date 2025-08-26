import matplotlib.pyplot as plt

from backtesting import Backtest as Backtets_btModule


class Backtest(Backtets_btModule):
    def __init__(self,
                 data,
                 strategy,
                 *,
                 cash=10000,
                 spread=0.0,
                 commission=0.0,
                 margin=1.0,
                 trade_on_close=False,
                 hedging=False,exclusive_orders=False,
                 finalize_trades=False):
       
        super().__init__(data,strategy,cash=cash,spread=spread,commission=commission,margin=margin,trade_on_close=trade_on_close,hedging=hedging,exclusive_orders=exclusive_orders,finalize_trades=finalize_trades)

    @property
    def stats(self):
        s = self.run()
        print(s)
        return s

    # Call plt.show() afterwards
    def plotEquityCurve(self):
        if self._results is not None:
            plt.plot(self._data.index,self._results["_equity_curve"]
            ["Equity"],alpha=0.65)
            plt.xlabel("Date")
            plt.ylabel("Equity")
            plt.title("Equity curve")
        else: 
            raise ValueError("Need to do bt.run() to obtain the equity curves")
        
