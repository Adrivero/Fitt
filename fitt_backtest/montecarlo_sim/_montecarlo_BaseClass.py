import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports


class Montecarlo():
    '''
    Allows the execution of a Montecarlo simulation of a trading strategy
    '''
    def __init__(self,data):
        self.data = data
        self.random_prices = None
        
    def generateRandomData(self,sim_days):
        self.random_prices = np.random.choice(self.data.values.flatten(), size=sim_days, replace=True)
        
   
        

