import pandas as pd
import vectorbt as vbt

start = "2016-01-01 UTC"
end = "2020-01-01 UTC"
prices = vbt.YFData.download(["META"],start=start,end=end).get("Close")

fast_ma = vbt.MA.run(prices, 10, short_name="fast")
slow_ma = vbt.MA.run(prices, 30, short_name="slow")

entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)
