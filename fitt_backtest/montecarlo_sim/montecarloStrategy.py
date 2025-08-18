from fitt_backtest.montecarlo_sim._montecarlo_BaseClass import Montecarlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    def __init__(self, data,strategy,risk_free_rate:float,):
        super().__init__(data)
        self.strategy = strategy
        self.risk_free_rate = risk_free_rate
        

        # Lazy initialized atributes
        self.starting_equity = None
        self.n_periods = None

    def simulate_RandomTrades(self,starting_equity = 10000,n_trades = None, n_simulations = 100,comissions = 0,n_periods = 252, plot= 0)->list:
        '''
        Don't change n_trades if your strategy has a time dependency (if it depends on a certain market regime)

        Parameters:
        n_trades(int): Change if you want to randomly pick more trades instead of the amount given by the initial backtest
        Returns: 
        simulations (list): List containing the possible equity curves 
        n_periods (int) : If using Timestamped data, the amount of periods, if using trades, n_trades total_trades/number_years

        Returns: 
        simulations (list): Contains the equity curve for each simulation
        '''

        # Initializing lazy atributes
        self.starting_equity = starting_equity
        self.n_periods = n_periods

        bt = Backtest(self.data,self.strategy,commission=comissions)
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
    
    # Just simualtes prices! Only valid for daily strategies
    def simulate_RandomWalk(self,starting_equity = 10000, periods_to_simulate = 252, n_simulations = 100, time_step = 1, comissions = 0, plot_random_paths=False, plot_equity_curves = False) -> list:
        """
        Simulates the asset price at time t using the Geometric Brownian Motion (GBM) model.

        Mathematical model: Comes from solving the stochastic equation dS_t = μ * S_t * dt + std * S_t * dW_t using Ito's Lemma
            S_t = S_{t-1} * exp((μ - 0.5 * std²) * Δt + std * sqrt(Δt) * Z)

        Where:
            - S_{t-1} : Asset price at the previous time step
            - μ       : Average return (drift), estimated from historical log returns
            - std       : Historical volatility of the asset
            - Δt      : Time step size (typically 1 day)
            - Z       : Stochastic component (Z ~ N(0, 1))

        Limitations:
            - Assumes constant μ and σ over time.
            - Does not capture extreme market events (jumps) or time-varying volatility.
        """

        print(f"Montecarlo_Strategy: (Random walk data) Starting to simulate n_simulations: {n_simulations} ")

        # Initializing lazy atributes
        self.starting_equity = starting_equity


        prices = self.data["Close"]
        last_date = self.data.index[-1]
        
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Parameters of our random walk model
        mu = log_returns.mean()
        std = log_returns.std()

        S0 = prices.iloc[-1] # Last known price 
        T = periods_to_simulate
        M = n_simulations

        # Columns: Simulations ; Rows: One moment in time
        simulated_paths = np.zeros((T,M))
        simulated_paths[0] = S0
        

        for t in range(1,T):
            z = np.random.standard_normal(M)
            simulated_paths[t] = simulated_paths[t-1] * np.exp((mu - 0.5 * std**2) * time_step + std * np.sqrt(time_step) * z)

        # Plotting price paths
        if plot_random_paths==True:
            for i in range(n_simulations):
                plt.plot(simulated_paths[:,i])
            

            plt.ylabel("Price")
            plt.xlabel("Days into the future")
            plt.title("Geometric Brownian Motion price paths")
            plt.show()


        close_prices = [simulated_paths[:, t].tolist() for t in range(simulated_paths.shape[1])]
        stats = []

        # Adding new dateTimeIndexes starting from the last data dat
        freq = pd.infer_freq(self.data.index) # Get frequency of the input data
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                    periods=periods_to_simulate,
                                    freq=freq)

        # The Backtest will actualyl start 2 days after the last given date due to the elimination of NaN values
        for i in range(n_simulations): 
            #Iterate across all different price curves
            data_df = pd.DataFrame()
            data_df["Close"] = close_prices[i] 
            data_df["Open"] = data_df["Close"].shift(1)
            data_df = data_df[["Open","Close"]]
            data_df.index = future_dates
            data_df.dropna(inplace=True)

            # Adding high and low columns
            data_df["High"] = data_df["Close"]
            data_df["Low"] = data_df["Open"]
            
            bt = Backtest(data = data_df, strategy=self.strategy, cash=starting_equity,commission=comissions)
            bt_results = bt.run()
            stats.append(bt_results)
            print(f"> Simulation number {i} ended - Sharpe: {bt_results["Sharpe Ratio"]}")
            
            if plot_equity_curves == True:
                bt.plotEquityCurve()
                plt.show()

        return stats
            

      
    