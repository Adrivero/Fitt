from fitt_backtest.montecarlo_sim._montecarlo_BaseClass import Montecarlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Union, List
import tqdm
from typing import Type
import inspect

# Local imports
from fitt_backtest.engine import Backtest
from utils.console_utils import cleanConsole
from fitt_metrics.statistics import Stats
from fitt_strategies.strategy_baseclass import Strategy

# Needs to be in the global scope for ProcessPoolExecutor to work (i.e to be pickable)
def _backtestEquity_curve(equity_curve, future_dates, strategy, starting_equity, commissions,margin,spread):
    data_df = pd.DataFrame()
    data_df["Close"] = equity_curve
    data_df["Open"] = data_df["Close"].shift(1)
    data_df = data_df[["Open", "Close"]]
    data_df.index = future_dates
    data_df.dropna(inplace=True)

    data_df["High"] = data_df["Close"]
    data_df["Low"] = data_df["Open"]

    bt = Backtest(data=data_df, strategy=strategy, cash=starting_equity, commission=commissions,margin=margin,spread = spread)
    return bt.run()

class Montecarlo_Strategy(Montecarlo):
    '''
    Used to perform Montecarlo simulation of a trading strategy
    '''
    def __init__(self, data: Type[pd.DataFrame],strategy:Type[Strategy],risk_free_rate:float = 0.02,starting_equity = 1000, comissions = 0,spread = 0,margin = 0):
        super().__init__(data)
        self.strategy = strategy

        self.risk_free_rate = risk_free_rate  # To calculate metrics

        self.starting_equity = starting_equity
        self.comissions = comissions
        self.spread = spread
        self.margin = margin

        # Lazy initialized atributes
        self.n_periods = None

    def simulate_RandomTrades(self,n_trades = None, n_simulations = 100,n_periods = 252, plot= 0)->list:
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
        self.n_periods = n_periods

        bt = Backtest(self.data,self.strategy,commission=self.comissions,margin=self.margin,spread=self.spread,cash=self.starting_equity)
        stats = bt.run() # <class 'backtesting._stats._Stats'>
        
        trade_returns = stats['_trades']['ReturnPct']
        
        # If n_trades are not given, use the number of trades given by the backtest
        if not n_trades:
            n_trades = len(trade_returns)

        simulations = [] # Contains the equity curve vealue for each simulation
        print(f"Montecarlo_Strategy: (Random picked trades) Starting to simulate n_simulations: {n_simulations} ")
        for _ in range(n_simulations):
            sample = np.random.choice(trade_returns, size=n_trades, replace=True)
            equity = pd.Series(sample).add(1).cumprod()*self.starting_equity # Sums 1 to create growth factors and calculates a new equity curve
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
    
    
    def simulate_RandomWalk(self, periods_to_simulate = 252, n_simulations = 100, time_step = 1, plot_random_paths=False, plot_equity_curves = False) -> list:
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

        Example:

            data = yf.download("AAPL",period="3y")
            data.columns = data.columns.get_level_values(0)
            mt = Montecarlo_Strategy(data=data,strategy=BBandsCrossing,risk_free_rate=0.02)
            trades = mt.simulate_RandomWalk(n_simulations=1000,periods_to_simulate=1000)
            mt.analysis(stats=trades)

        """

        print(f"Montecarlo_Strategy: (Random walk data) Starting to simulate n_simulations: {n_simulations} ")

        
        

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

        # Adding new dateTimeIndexes starting from the last data data
        freq = pd.infer_freq(self.data.index) # Get frequency of the input data
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                    periods=periods_to_simulate,
                                    freq=freq)
        import time
        start_time = time.perf_counter()

        # Use multiprocessing to compute each backtest for all all equity curves
        with ProcessPoolExecutor() as executor:
              results = executor.map(
                _backtestEquity_curve,
                close_prices,
                [future_dates] * len(close_prices),
                [self.strategy] * len(close_prices),
                [self.starting_equity] * len(close_prices),
                [self.comissions] * len(close_prices),
                [self.margin]* len(close_prices),
                [self.spread]*len(close_prices)
            )

        duration = time.perf_counter() - start_time
        cleanConsole()
        print("Simulation duration: ", duration)
        print(f"n_simulations: {n_simulations}")
        print(f"Periods simulated: {periods_to_simulate}")
        return list(results)

    
    
    
    # ------------------------------------- Monte Carlo Permutation Testing (MCPT)------------------------------------- 
    #TODO Continue implementing the walk-forward and insample testing...
    
    @staticmethod
    def _get_permutation(ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None):
        '''
        In-Sample permutation tests
        Mantains the distribution of the real movement, including Skew, Kurtosis and volatility changes
        '''
        assert start_index >= 0
        

        np.random.seed(seed)

        if isinstance(ohlc, list):
            time_index = ohlc[0].index
            for mkt in ohlc:
                assert np.all(time_index == mkt.index), "Indexes do not match"
            n_markets = len(ohlc)
        
        else:    
            n_markets = 1
            time_index = ohlc.index
            ohlc = [ohlc]

        n_bars = len(ohlc[0])

        perm_index = start_index + 1
        perm_n = n_bars - perm_index

        start_bar = np.empty((n_markets, 4))
        relative_open = np.empty((n_markets, perm_n))
        relative_high = np.empty((n_markets, perm_n))
        relative_low = np.empty((n_markets, perm_n))
        relative_close = np.empty((n_markets, perm_n))

        for mkt_i, reg_bars in enumerate(ohlc):
            log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']])

            # Get start bar
            start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

            # Open relative to last close
            r_o = (log_bars['Open'] - log_bars['Close'].shift()).to_numpy()
            
            # Get prices relative to this bars open
            r_h = (log_bars['High'] - log_bars['Open']).to_numpy()
            r_l = (log_bars['Low'] - log_bars['Open']).to_numpy()
            r_c = (log_bars['Close'] - log_bars['Open']).to_numpy()

            relative_open[mkt_i] = r_o[perm_index:]
            relative_high[mkt_i] = r_h[perm_index:]
            relative_low[mkt_i] = r_l[perm_index:]
            relative_close[mkt_i] = r_c[perm_index:]

        idx = np.arange(perm_n)

        # Shuffle intrabar relative values (high/low/close)
        perm1 = np.random.permutation(idx)
        relative_high = relative_high[:, perm1]
        relative_low = relative_low[:, perm1]
        relative_close = relative_close[:, perm1]

        # Shuffle last close to open (gaps) seprately
        perm2 = np.random.permutation(idx)
        relative_open = relative_open[:, perm2]

        # Create permutation from relative prices
        perm_ohlc = []
        for mkt_i, reg_bars in enumerate(ohlc):
            perm_bars = np.zeros((n_bars, 4))

            # Copy over real data before start index 
            log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']]).to_numpy().copy()
            perm_bars[:start_index] = log_bars[:start_index]
            
            # Copy start bar
            perm_bars[start_index] = start_bar[mkt_i]

            for i in range(perm_index, n_bars):
                k = i - perm_index
                perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
                perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
                perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
                perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

            perm_bars = np.exp(perm_bars)
            perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['Open', 'High', 'Low', 'Close'])

            perm_ohlc.append(perm_bars)

        if n_markets > 1:
            return perm_ohlc
        else:
            return perm_ohlc[0]

    
    @staticmethod
    def sanitize_opt_kwargs(**kwargs):
        """
        Validates that the arguments passed exist in bt.optimize().
        Raises ValueError if it finds any invalid ones.
        """
        sig = inspect.signature(Backtest.optimize)
        valid_params = set(sig.parameters.keys())

        for k in kwargs:
            if k not in valid_params:
                raise ValueError(
                    f"Invalid argument for bt.optimize(): '{k}'. "
                    f"Allowed parameters: {sorted(valid_params)}"
                )
        return kwargs

    def in_sample_mcpt(self,n_permutations = 1000, metric = "Profit Factor",starting_index = 0, **opt_kwargs):

        initial_bt = Backtest(
            self.data,
            self.strategy,
            cash=self.starting_equity,
            margin=self.margin,
            spread=self.spread,
            commission=self.comissions
        )

       # In-Sample historical (Supposedly better metric)
        best_bt = Backtest(data=self.data,
                           strategy=self.strategy,
                           cash=self.starting_equity,
                           spread=self.spread,
                           commission=self.comissions,
                           margin=self.margin
                           )

        opt_kwargs = self.sanitize_opt_kwargs(**opt_kwargs)
        best_stats = best_bt.optimize(**opt_kwargs)

        best_metric = best_stats[metric] # Assumed best metric obtained from historical data

        # Backtesting permutated data
        
        perm_better_count = 1 # Number of times the optimizated strategy for a specifi permutation behaves better than insample data
        permutated_metrics = []
        
        print("Starting in-sample MCPT")
        for perm_i in tqdm(range(1,n_permutations)):
            training_data = self._get_permutation(self.data,start_index=starting_index)

            bt = Backtest(data=self.data,
                           strategy=self.strategy,
                           cash=self.starting_equity,
                           spread=self.spread,
                           commission=self.comissions,
                           margin=self.margin
                           )
            
            stats = bt.optimize(**opt_kwargs)
            
            if stats[metric]>= best_metric:
                perm_better_counter+=1

            permutated_metrics.append(stats[metric])

        insample_mcpt_pval = perm_better_counter/n_permutations
        print(f"In-Sample MCPT P_Value: {insample_mcpt_pval}")

        plt.style.use("dark_background")
        pd.Series(permutated_metrics).hist(label="Permutations")
        plt.axvline(best_metric,color="Red",label="In sample PF")
        plt.xlabel(metric)
        plt.title(f"In-Sample MCPT. P-value {insample_mcpt_pval}")
        plt.legend()
        plt.show()








   

      
    