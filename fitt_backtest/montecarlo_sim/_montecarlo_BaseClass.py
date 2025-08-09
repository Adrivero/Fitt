import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from scipy.stats import norm

from backtesting._stats import _Stats 
# Local imports
from fitt_metrics.statistics import Stats

class Montecarlo():
    '''
    Allows the execution of a Montecarlo simulation of a trading strategy
    '''
    def __init__(self,data):
        self.data = data
        self.random_prices = None
        

      

    #TODO Implement those analysis for all equity curves at the same time:
    # Median & mean final return – central tendency.

    # Worst-case and best-case outcomes – tail risks & upside.

    # Percentiles – e.g., 5th percentile tells you "there’s a 95% chance of doing better than this."

    # Probability of loss – how often the strategy ends negative.

    # Distribution of max drawdowns – what’s the likely pain you’ll endure.

    # Risk-of-ruin – probability of the portfolio going below a critical threshold (e.g., -50% or zero).

    # Skewness & kurtosis – are results asymmetric or prone to extreme outliers?


    def calculateStatsFromEquityCurves(self,eq_curves: list)->pd.DataFrame:
        sharpes = []
        max_drawdowns = []
        profit_factors = []
        statitsitcs = []

        for sim in eq_curves: 
            st = Stats(sim,self.risk_free_rate,self.n_periods)
            st_df = st.compute_Stats()
            
            sharpe = st_df.loc["Sharpe ratio", "Value"]
            max_dr = st_df.loc["Max. Drawdown", "Value"]
            profit_factor = st_df.loc["Profit factor", "Value"]

            sharpes.append(sharpe)
            max_drawdowns.append(max_dr)
            profit_factors.append(profit_factor)

            statitsitcs.append(st)

        # Eliminate NaNs and Infinities
        sharpes = [x for x in sharpes if not (np.isnan(x) or np.isinf(x))]
        max_drawdowns = [x for x in max_drawdowns if not (np.isnan(x) or np.isinf(x))]
        profit_factors = [x for x in profit_factors if not (np.isnan(x) or np.isinf(x))]

    
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        
        axs[0].hist(sharpes,bins=20,density = True)
        axs[0].set_title("Sharpes")
        axs[0].set_ylabel("Frequency")
    
        mu = np.mean(sharpes)
        sigma = np.std(sharpes)

        x = np.linspace(min(sharpes), max(sharpes), 1000)
        y = norm.pdf(x, loc=mu, scale=sigma)

        axs[0].plot(x, y, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
        axs[0].legend()

        axs[1].hist(max_drawdowns,bins=20)
        axs[1].set_title("Max_drawdowns")
        axs[1].set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    
        return_dict = {
            "Worst sharpe" : min(sharpes),
            "Mean sharpe" : np.mean(sharpes),
            "Best sharpe" : max(sharpes),
            "Worst Max Drawdown" : min(max_drawdowns),
            "Mean Max Drawdown" : np.mean(max_drawdowns),
            "Best Max Drawdown" : max(max_drawdowns),
            "Worst Profit factor" : min(profit_factors), 
            "Mean Profit factor" : np.mean(profit_factors),
            "Best Profit factor" : max(profit_factors)
        }
        return_df = pd.Series(return_dict,name="Value").to_frame()
        print(return_df)
        return return_df

    def analysis(self,equity_curves:List[list] = None, stats: List[_Stats]= None  ) ->list:
        if equity_curves is None and stats is None:
            raise ValueError("(Equity_curves) or (_Stats list) missing")
        if equity_curves and stats:
            raise ValueError ("Expected only one simulation data, equity_curves or _stats list")
        

        if equity_curves:
            self.calculateStatsFromEquityCurves(equity_curves)
            
        if stats:
            # TODO in future, transform this into the same function as above so you get the same results for both types of montecarlo
            sharpes =[]
            returns_pct = []

            for s in stats:
                s["Return [%]"]
                s["Return (Ann.) [%]"]
                s["Sharpe Ratio"]
                s["Max. Drawdown [%]"]
                s["Avg. Drawdown [%]"]

