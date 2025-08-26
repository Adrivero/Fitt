import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np



def resistance_byVolume(data,num_bins,n):
    '''
    Parameters:
    data (DataFrame): Contains the desired OHLCV data
    num_bins (int): Number of price steps in which to divide the whole dataset
    n (int): Number of price-intervals to be considered as resistance (taking the top ones is usually the best option)

    Returns:
    resistancePrice (list): All price resistance found for the given configuration

    Volume-by-Price Resistance Detection 

    This strategy identifies potential resistance levels by analyzing where
    the highest trading volumes occurred across historical price ranges.

    Concept:
    Resistance levels are prices where the stock has difficulty moving higher.
    They often form where many traders have previously bought or sold,
    and are now waiting to exit or re-enter positions.

    This method analyzes volume traded at different price levels (not time),
    creating a horizontal volume profile.

    How it works:
    1. The full price range is split into evenly spaced intervals ("bins").
    2. Each historical price is assigned to a bin based on its closing price.
    3. Total volume traded in each bin is summed.
    4. The bins with the highest volume are likely resistance levels.

    Why:
    - High-volume price levels are areas of high interest and liquidity.
    - They tend to act as magnets for price and often cause stalls or reversals.
    - If price is below one of these levels, it may face resistance there.
    '''

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("Given data is not a DataFrame")
    
    # Prepare the price range bins
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), num_bins)

    # Assign each closing price to a price bin
    df['PriceBin'] = pd.cut(df['Close'], bins=price_bins)

    # Sum the volume for each price bin
    volume_by_price = df.groupby('PriceBin')['Volume'].sum()

    # Get the center price of each bin for plotting
    bin_centers = [interval.mid for interval in volume_by_price.index]

    top_resistances = sorted(bin_centers,reverse=True)[:n]
    
    return top_resistances

