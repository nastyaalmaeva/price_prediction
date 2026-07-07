import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

os.makedirs('../plots', exist_ok=True)

df = pd.read_csv('../data/stock_data.csv')

if 'Price' in df.columns:
    df.drop(columns=['Price'], inplace=True)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, errors='coerce')

print('dataset shape:', df.shape)
print('\ncolumns:', df.columns.tolist())
print('\nindex sample:', df.index[:5])
print('\nmissing values:\n', df.isnull().sum())
print('\nstatistics:\n', df.describe().round(2))

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.title('Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.savefig('../plots/close_price.png', dpi=300)
plt.close()

df['returns'] = df['Close'].pct_change()
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['returns'], label='Daily Returns', color='green', alpha=0.7)
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.savefig('../plots/daily_returns.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(df['returns'].dropna(), bins=50, kde=True, color='green')
plt.title('Distribution of Daily Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig('../plots/returns_distribution.png', dpi=300)
plt.close()

df['year'] = df.index.year
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='year', y='Close')
plt.title('Close Price Distribution by Year')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.savefig('../plots/boxplot_by_year.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Matrix')
plt.savefig('../plots/correlation_matrix.png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(df['Close'].dropna(), ax=axes[0], lags=40)
plot_pacf(df['Close'].dropna(), ax=axes[1], lags=40)
plt.savefig('../plots/acf_pacf.png', dpi=300)
plt.close()

decomposition = seasonal_decompose(df['Close'].dropna(), model='additive', period=30)
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], legend=False)
axes[0].set_title('Observed')
decomposition.trend.plot(ax=axes[1], legend=False)
axes[1].set_title('Trend')
decomposition.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_title('Seasonal')
decomposition.resid.plot(ax=axes[3], legend=False)
axes[3].set_title('Residual')
plt.tight_layout()
plt.savefig('../plots/decomposition.png', dpi=300)
plt.close()
