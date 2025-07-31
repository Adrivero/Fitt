from fitt_backtest.montecarlo_sim._montecarlo_BaseClass import Montecarlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Local imports
from fitt_backtest.engine import Backtest
from utils.console_utils import cleanConsole

# TODO Add Montecarlo for randomly generated data 
# TODO Add Montecarlo for randomly picked trades

class MontecarloStrategy(Montecarlo):
    '''
    Used to perform Montecarlo simulation of a trading strategy
    '''
    def __init__(self, data,strategy):
        '''
        Parameters:

        Returns:

        '''
        super().__init__(data)
        self.strategy = strategy

    def simulate_RandomTrades(self,starting_equity = 1000, n_simulations = 100,plot = 1)->list:
        '''
        Parameters:

        Returns:
        '''
        bt = Backtest(self.data,self.strategy)
        stats = bt.run() # <class 'backtesting._stats._Stats'>
        
        trade_returns = stats['_trades']['ReturnPct']

        simulations = [] # Contains the equity curve vealue for each simulation

        for _ in range(n_simulations):
            sample = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
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
    
    



