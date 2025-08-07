import pandas as pd
import numpy as np


class Stats():
    def __init__(self,equity_curve,risk_free_rate,n_periods):
        '''
        equity_curve (list): List containing all equity datapoints
        risk_free_rate (float) : Annualized risk_free_rate
        n_periods(int) Total number of time periods that the trading strategy was active
        '''
        self.equity_curve = equity_curve
        self.risk_free_rate = risk_free_rate
        self.n_periods = n_periods

        # All of the ratios and their calcuulation
        self.ratios_dict ={
                       "Max. Drawdown" : self.max_drawdown(self.equity_curve),

                       "Avg. Drawdown" : self.avg_drawdown(self.equity_curve),

                       "Sharpe ratio" : self.sharpe_ratio_from_equity(self.equity_curve,self.risk_free_rate,self.n_periods),

                       "Profit factor" : self.profit_factor_from_equity(self.equity_curve),

                       "Expectancy" : self.expectancy(self.equity_curve)

                       }

    # --------------- Ratios------------------------------------
    @staticmethod
    def drawdowns(equity_curve):
        cum_max = equity_curve.cummax()
        drawdown = equity_curve / cum_max - 1
        return drawdown

    @staticmethod
    def avg_drawdown(equity_curve):
        dd = Stats.drawdowns(equity_curve)
        dd_periods = dd[dd < 0]
        return dd_periods.mean() * 100 
    
    
    # drawdown_t = (equity_t / max_acumulated till t) - 1
    @staticmethod
    def max_drawdown(equity_curve: pd.Series):
        drawdown = Stats.drawdowns(equity_curve)
        return drawdown.min()
    

    # Expectancy=(Pw​×Aw​)−(Pl​×Al​) 
    @staticmethod
    def expectancy(equity_curve: pd.Series):
        returns = equity_curve.pct_change().dropna()
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        P_w = len(wins) / len(returns)
        P_l = 1 - P_w

        A_w = wins.mean()
        A_l = abs(losses.mean())  # ensure it's positive

        return (P_w * A_w) - (P_l * A_l)

    # mean(R-Rf)/std(R)
    @staticmethod
    def sharpe_ratio_from_equity(equity_curve: pd.Series, risk_free_rate=0, periods_per_year=252):
        '''
        Calculate the annualized Sharpe ratio from an equity curve.

        Note:
        -----
        If given a trade-level equity curve n_periods = total_trades / total_years (mean number of trades per year)
        When using trade-level equity curves, ensure periods_per_year reflects the average 
        number of trades per year, not the number of trading days, to avoid overstating Sharpe.

        Parameters:
        ----------
        equity_curve : pd.Series
            A pandas Series representing the equity value over time 

        risk_free_rate : float, optional (default=0)
            The annualized risk-free rate (e.g., 0.03 for 3%). This will be adjusted
            to the frequency of the equity_curve using the periods_per_year argument.

        periods_per_year : int, optional (default=252)
            Number of periods in a year. Use:
                - 252 for daily returns
                - 52 for weekly returns
                - 12 for monthly returns
                - Avg. trades/year for trade-level returns

        Returns:
        -------
        sharpe : float
            The annualized Sharpe ratio, measuring the strategy's risk-adjusted return.


        Formula:
        --------
            Sharpe = sqrt(periods_per_year) * mean(returns - rf_per_period) / std(returns)

        
        '''
        returns = equity_curve.pct_change().dropna()

        # Adapt risk_free_rate (annualized) to a per_period basis
        rf_per_period = ((1 + risk_free_rate) ** (1/periods_per_year) )- 1

        excess_returns = returns - rf_per_period

        # Annualized sharpe ratio
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

        return sharpe
    
    # gains/losses
    @staticmethod
    def profit_factor_from_equity(equity_curve : pd.Series):
        returns = equity_curve.pct_change().dropna()
        gains = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()

        if losses == 0:
            return float('inf')  
        
        return gains / abs(losses)


    # Returns DataFrame with all the metrics
    def compute_Stats(self):
        df = pd.Series(self.ratios_dict, name="Value").to_frame()

        if df.isna().any().any():
            print("Runtime Warning -> NaN and/or Inf found when computing the stats DataFrame")
        return df

    
    # ----------------- Printing --------------------------------

    def printStats(self):
        df = self.compute_Stats()
        print(df)

    def __repr__(self):
        self.printStats()
        return "\n"