
# ``Fitt`` 

<div align="center">
  <img width="300" height="300" alt="Fitt logo square" src="https://github.com/user-attachments/assets/90327620-8d98-4719-8222-46970615ec02"/>
</div>

Fully integrated trading tool (Fitt) aims to be the only suite necessary for developing, testing and deploying new strategies.



# Table of contents
1) [Strategies module](#1-strategies-module): Severeal default strategies (ML, Quant, Technical Analysis...) already included. You can add your new strategies or modify current ones.
2) [Backtesting And Montecarlo module](#2-backtesting-and-montecarlo-module): Perform backtesting with historical data or Montecarlo simulation of the different trading strategies.

</br>

# 1. Strategies module 
Severeal strategies come already implemented
- Rule-based Strategies (e.g EMA crossings, Bollinger-Bands, mean-reversion...)
- Quantitative Strategies  (TBI)
- Pairs trading and arbitrage (TBI)
- Machine Learning Strategies

</br>

# 2. Backtesting And Montecarlo module
- **Purpose**: Get an accurate representation of the trading strategies performance
Used to test different already implemented strategies or to test new ones.

#### Quick example
```python
import yfinance as yf

from fitt_strategies.simpleCrosses.bollingerBandsCrossing import BBandsCrossing

from fitt_backtest.engine import Backtest

data = yf.download("AAPL",start="2020-01-01")
data.columns = data.columns.get_level_values(0)

bt = Backtest(data=data,strategy=BBandsCrossing, cash=10000, commission=0.02)
output = bt.run()
bt.plot()
```

#### Output:
```
Start                     2020-01-02 00:00:00
End                       2025-08-11 00:00:00
Duration                   2048 days 00:00:00
Exposure Time [%]                    40.38325
Equity Final [$]                   8413.21422
Equity Peak [$]                    15095.3844
Commissions [$]                   10022.13104
Return [%]                          -15.86786
Buy & Hold Return [%]               190.44124
Return (Ann.) [%]                    -3.04294
Volatility (Ann.) [%]                16.58819
CAGR [%]                             -2.10358
Sharpe Ratio                         -0.18344
Sortino Ratio                        -0.25044
Calmar Ratio                         -0.06792
Alpha [%]                           -63.30471
Beta                                  0.24909
Max. Drawdown [%]                   -44.80279
Avg. Drawdown [%]                    -5.73861
Max. Drawdown Duration     1805 days 00:00:00
Avg. Drawdown Duration      138 days 00:00:00
# Trades                                   25
Win Rate [%]                             60.0
Best Trade [%]                       29.22639
Worst Trade [%]                      -7.08869
Avg. Trade [%]                        3.30318
Max. Trade Duration         100 days 00:00:00
Avg. Trade Duration          32 days 00:00:00
Profit Factor                         3.64385
Expectancy [%]                        3.62243
SQN                                   2.00438
Kelly Criterion                       0.41963
_strategy                      BBandsCrossing
_equity_curve                             ...
_trades                       Size  EntryB...
```

<div align="center">
<img width="1919" height="665" alt="Backtest result" src="https://github.com/user-attachments/assets/93528b19-e6f9-486e-bc78-f38c8278e9f3" />
</div>


## 2-.1 Montecarlo simulation
- Random historical data (Random-walk): A Geometric Brownian Motion creates various price paths. Then the trading strategy will be tested in those "random" price paths and and the main mean metrics will be computed.
- Random Trade sampling with bootstraping. From a collection of trades, choose trades randomly with replacement.
  </br>
  
  <div align="center">
  <img width="540" height="380" alt="Geometric Brownian Motion paths" src="https://github.com/user-attachments/assets/b2a6a804-b9bf-44f6-86fc-3e217879b91e" />
  </div>
  
</br>

#### Quick example

``` python
from fitt_backtest.montecarlo_sim.montecarloStrategy import Montecarlo_Strategy
from fitt_strategies.simpleCrosses.bollingerBandsCrossing import BBandsCrossing

data = yf.download("AAPL",start="2020-01-01")
data.columns = data.columns.get_level_values(0)

mt = Montecarlo_Strategy(data,BBandsCrossing,risk_free_rate=0.2)
simulations_stats = mt.simulate_RandomPath(periods_to_simulate=252,n_simulations=100)

output = mt.analysis(stats=simulations_stats)
```

#### Output
<div align="center">
<img width="1193" height="489" alt="Sharpe and Max DDS Distributions" src="https://github.com/user-attachments/assets/1ad68861-ac2e-4616-8a9c-7e3426c5f448" />
</div>
</br>

```
                                                  Simulation results
Probability of loss [%]                                         46.0
Probability of Ruin [%] (Portfolio losing >30.0%)                0.0

Avg Buy & Hold returns [%]                                    15.851
Return Worst [%]                                             -25.577
Return Mean [%]                                                4.937
Return Best [%]                                               90.451
Return P5 [%]                                                -21.156
Return P50 [%]                                                 1.655
Return P95 [%]                                                32.759
Ann. Return Mean [%]                                           8.306
Skewness (Returns)                                             1.446
Pearson Kurtosis (Includes normal baseline) (Returns)           7.01

Sharpe Worst                                                  -3.895
Sharpe Mean                                                    -0.02
Sharpe Best                                                    2.397
Sharpe P5                                                     -2.326
Sharpe P50                                                     0.117
Sharpe P95                                                     1.358

MaxDD Worst [%]                                              -29.746
MaxDD Mean [%]                                               -14.592
MaxDD Best [%]                                                  -6.1
MaxDD P5 [%]                                                 -22.759
MaxDD P50 [%]                                                -14.235
MaxDD P95 [%]                                                 -7.851
Avg. Drawdown Mean [%]                                         -7.74

Skewness >0, 'more' positive returns
Kurtosis >3, Fat Tails, more probability of extreme events
```

</br>


# 3. Deployment module (To be implemented)

Allows the deployment of strategies with different APIS (IBKR API, Alpaca, Robinhood)

</br>

# 4. Risk module (To be implemented)
Used to obtain the different risk metrics

</br>


# 5. Portfolio management module (To be implemented)
Used to manage the different strategies and their impact on the whole portfolio

</br>


## License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](./LICENSE) file for more details.
