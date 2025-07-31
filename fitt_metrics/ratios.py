import numpy as np
import pandas as pd


# drawdown_t = (equity_t / max_acumulated till t) - 1
def max_drawdown(equity_curve: pd.Series):
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1
    return drawdown.min()

# mean(R-Rf)/std(R)
def sharpe_ratio_from_equity(equity_curve: pd.Series, risk_free_rate=0, periods_per_year=252):
    '''
    Calculate the annualized Sharpe ratio from an equity curve.

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
