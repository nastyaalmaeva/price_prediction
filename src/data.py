import yfinance as yf
import pandas as pd
import os

os.makedirs('../data', exist_ok=True)

ticker = 'AAPL'
df = yf.download(ticker, start='2012-01-01', end='2025-01-01')

df.index = pd.to_datetime(df.index)

df.to_csv('../data/stock_data.csv')

print(f'saved {len(df)} rows to data/stock_data.csv')
print(df.head())