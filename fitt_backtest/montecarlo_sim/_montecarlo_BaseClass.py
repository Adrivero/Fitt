import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from scipy.stats import norm
from scipy.stats import skew, kurtosis

from backtesting._stats import _Stats 
# Local imports
from fitt_metrics.statistics import Stats

class Montecarlo():
    '''
    Allows the execution of a Montecarlo simulation of a trading strategy
    '''
    def __init__(self,data):
        self.data = data
        

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

    def analysis(self,equity_curves:List [list] = None, stats: List[_Stats]= None, own_comments: bool = True ) ->list:

        if equity_curves is None and stats is None:
            raise ValueError("(Equity_curves) or (_Stats list) missing")
        if equity_curves and stats:
            raise ValueError ("Expected only one simulation data, equity_curves or _stats list")
        

        if equity_curves:
            self.calculateStatsFromEquityCurves(equity_curves)
            
        if stats:
            # TODO in the future, transform this into the same function as above so you get the same results for both types of montecarlo
            sharpes =[]
            returns_pct = []
            ann_returns_pct = []
            max_dds = []
            avg_dds = []
            avg_bnh_returns = []
            # Getting all the data for all the simulations
            for s in stats:
                returns_pct.append(s["Return [%]"])
                ann_returns_pct.append(s["Return (Ann.) [%]"])
                sharpes.append(s["Sharpe Ratio"])
                max_dds.append(s["Max. Drawdown [%]"])
                avg_dds.append(s["Avg. Drawdown [%]"])
                avg_bnh_returns.append(s["Buy & Hold Return [%]"])



            # Win and Ruin probability calculation
            win_prob = len([r for r in returns_pct if r > 0]) / len(returns_pct)

            ruin_threshold = -0.3 # Loss percentage
            ruin_prob = len([r for r in returns_pct if r <= ruin_threshold * 100]) / len(returns_pct)

            # kurtosis and skewness calculation
            kurt = kurtosis(returns_pct, fisher=False)
            skewness = skew(returns_pct)

            # The spaces in between have to be of different length as dictionaries have no repeated elements
            return_dict = {
            "Win Rate [%] (Probability of positive returns)": round(win_prob * 100, 3),
            f"Probability of Ruin [%] (Portfolio losing >{-ruin_threshold * 100}%)": round(ruin_prob * 100, 3),
            " ": " ",
            "Avg Buy & Hold returns [%]": round(np.mean(avg_bnh_returns), 3),
            "Return Worst [%]": round(min(returns_pct), 3),
            "Return Mean [%]": round(np.mean(returns_pct), 3),
            "Return Best [%]": round(max(returns_pct), 3),
            "Return P5 [%]": round(np.percentile(returns_pct, 5), 3),
            "Return P50 [%]": round(np.percentile(returns_pct, 50), 3),
            "Return P95 [%]": round(np.percentile(returns_pct, 95), 3),
            "Ann. Return Mean [%]": round(np.mean(ann_returns_pct), 3),
            "Skewness (Returns)": round(skewness, 3),
            "Pearson Gross Kurtosis (Includes normal baseline) (Returns)": round(kurt, 3),
            "Returns(Ann)/drawdown" : round(np.mean(ann_returns_pct)/np.mean(max_dds)),
            "  ": " ",

            "Sharpe Worst": round(min(sharpes), 3),
            "Sharpe Mean": round(np.mean(sharpes), 3),
            "Sharpe Best": round(max(sharpes), 3),
            "Sharpe P5": round(np.percentile(sharpes, 5), 3),
            "Sharpe P50": round(np.percentile(sharpes, 50), 3),
            "Sharpe P95": round(np.percentile(sharpes, 95), 3),
            "   ": "   ",

            "MaxDD Worst [%]": round(min(max_dds), 3),
            "MaxDD Mean [%]": round(np.mean(max_dds), 3),
            "MaxDD Best [%]": round(max(max_dds), 3),
            "MaxDD P5 [%]": round(np.percentile(max_dds, 5), 3),
            "MaxDD P50 [%]": round(np.percentile(max_dds, 50), 3),
            "MaxDD P95 [%]": round(np.percentile(max_dds, 95), 3),
            "Avg. Drawdown Mean [%]": round(np.mean(avg_dds), 3)
            }

            df = pd.Series(return_dict, name="Simulation results").to_frame()
            print("")
            print(df)

            print("")

            if own_comments:
                print("------------------ Automatic comments on the strategy performance ------------------")
                if skewness>0:
                    print("Skewness >0, 'more' positive returns ")
                elif skewness<0:
                    print("Skewness <0, 'more' negative returns ")
                else: print("Skewnes == 0, symetric")

                if kurt > 3 :
                    print("Kurtosis >3, Fat Tails, more probability of extreme events")
                
                ret_dd_ratio = round(np.mean(ann_returns_pct)/np.mean(max_dds))
                if ret_dd_ratio<2:
                    print("WARNING-Returns(Ann)/Drawdown is <2")

                if -ruin_threshold * 100 > 10:
                    print("WARNING-Risk of ruin higher than 10%")



        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Returns distribution
        axs[0].hist(returns_pct, bins=20, color='blue', alpha=0.7)
        axs[0].set_xlabel("Returns [%]")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("Returns [%] Distribution")
        axs[0].grid(True)

        # Max Drawdowns distribution
        axs[1].hist(max_dds, bins=20, color='orange', alpha=0.7)
        axs[1].set_xlabel("Drawdown [%]")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Max Drawdowns [%] Distribution")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
        
        return df