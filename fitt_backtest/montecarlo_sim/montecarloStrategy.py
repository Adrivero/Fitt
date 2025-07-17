from fitt_backtest.montecarlo_sim.montecarlo_BaseClass import Montecarlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MontecarloStrategy(Montecarlo):
    '''
    Used to perform Montecarlo simulation of a trading strategy
    '''
    def __init__(self, data,strategy,):
        '''
        Parameters:

        Returns:

        '''
        super().__init__(data)

    def simulate(self,n_simulations = 100, sim_days = 252):
        sampled_prices = np.random.choice(self.data.values.flatten(), size=sim_days, replace=True)
        
        pass