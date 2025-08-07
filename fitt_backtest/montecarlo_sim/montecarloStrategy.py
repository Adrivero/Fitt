from fitt_backtest.montecarlo_sim._montecarlo_BaseClass import Montecarlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import norm
import sys

# Local imports
from fitt_backtest.engine import Backtest
from utils.console_utils import cleanConsole
from fitt_metrics.statistics import Stats

# TODO Add Montecarlo for randomly generated data 

class Montecarlo_Strategy(Montecarlo):
    '''
    Used to perform Montecarlo simulation of a trading strategy
    '''
    def __init__(self, data,strategy,risk_free_rate:float,n_periods:int):
        '''
        Parameters:

        Returns:

        '''
        super().__init__(data)
        self.strategy = strategy
        self.risk_free_rate = risk_free_rate
        self.n_periods = n_periods

        # Lazy initialized atributes
        self.starting_equity = None


    def simulate_RandomTrades(self,starting_equity = 1000,n_trades = None, n_simulations = 100,plot = 0)->list:
        '''
        Don't change n_trades if your strategy has a time dependency (if it depends on a certain market regime)

        Parameters:
        n_trades(int): Change if you want to randomly pick more trades instead of the amount given by the initial backtest
        Returns: 
        simulations (list): List containing the possible equity curves 
        '''

        # Initializing lazy atributes
        self.starting_equity = starting_equity
        

        bt = Backtest(self.data,self.strategy)
        stats = bt.run() # <class 'backtesting._stats._Stats'>
        
        trade_returns = stats['_trades']['ReturnPct']
        
        # If n_trades are not given, use the number of trades given by the backtest
        if not n_trades:
            n_trades = len(trade_returns)

        simulations = [] # Contains the equity curve vealue for each simulation
        print(f"Montecarlo_Strategy: (Random picked trades) Starting to simulate n_simulations: {n_simulations} ")
        for _ in range(n_simulations):
            sample = np.random.choice(trade_returns, size=n_trades, replace=True)
            equity = pd.Series(sample).add(1).cumprod()*starting_equity # Sums 1 to create growth factors and calculates a new equity curve
            simulations.append(equity)
                    

        if plot == 1:
            plt.figure(figsize=(10, 6))
            for curve in simulations:
                plt.plot(curve)
            plt.title("Monte Carlo Simulation(Trades sampling with bootstrapping)")
            plt.xlabel("Trade #")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Display only integers in xlabel 

            plt.ylabel("Equity curve")
            plt.grid(True)
            plt.show()
        
        return simulations
    
    
    def analysis(self,simulations:list) ->list:
        '''
        simulations(list) : List including all lists trades
        '''

        sharpes = []
        max_drawdowns = []
        profit_factors = []
        statitsitcs = []

        for sim in simulations: 
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