#Importing Libraries
import os
import ta
import pandas as pd
import matplotlib.pyplot as plt

#Importing data into df form from csv form
df = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\LuxDev DSA\Capstone-Project\Crypto-BuySell-Model\notebooks\data\processed\BTCUSDT_1d.csv")
df.info()
df.head()

#Step 3 --- Feature Engineering
#Returns
df['return_1d'] = df['close'].pct_change(1)
df['return_1d']

df['return_7d'] = df['close'].pct_change(7)
df['return_7d']

df["rolling_volatility"] = df["close"].pct_change().rolling(7).std()
df.info()

#Technical Indicators
#RSI Indicator
df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

#Simple Moving Averages - SMA20, SMA50, SMA200
df["sma_20"] = df["close"].rolling(20).mean()
df["sma_50"] = df["close"].rolling(50).mean()
df["sma_200"] = df["close"].rolling(200).mean()
df.info()

#Bollinger Bands with a Histogram plot for visualization
# Importing BollingerBands submodule from ta.volatility
from ta.volatility import BollingerBands

# Calculating Bollinger Bands
bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)

df['bb_upper'] = bb.bollinger_hband()
df['bb_middle'] = bb.bollinger_mavg()
df['bb_lower'] = bb.bollinger_lband()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='Close Price', linewidth=2)
plt.plot(df.index, df['bb_upper'], label='Upper Band', linestyle='--', alpha=0.7)
plt.plot(df.index, df['bb_middle'], label='Middle Band (SMA)', linestyle='--', alpha=0.7)
plt.plot(df.index, df['bb_lower'], label='Lower Band', linestyle='--', alpha=0.7)
plt.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
plt.title('Bollinger Bands')
plt.legend()
plt.show()

# MACD with a Histogram plot for visualization
# Calculating MACD and adding it as new features to the DataFrame
macd_indicator = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)

df['macd'] = macd_indicator.macd()
df['macd_signal'] = macd_indicator.macd_signal()
df['macd_histogram'] = macd_indicator.macd_diff()

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Price
ax1.plot(df.index, df['close'], label='Close Price', linewidth=2)
ax1.set_ylabel('Price')
ax1.set_title('Price Chart')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MACD
ax2.plot(df.index, df['macd'], label='MACD Line', linewidth=2, color='blue')
ax2.plot(df.index, df['macd_signal'], label='Signal Line', linewidth=2, color='red')

# Histogram with colors (green = positive, red = negative)
colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
ax2.bar(df.index, df['macd_histogram'], label='Histogram', color=colors, alpha=0.3)

# Add zero line
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax2.set_ylabel('MACD')
ax2.set_xlabel('Date')
ax2.set_title('MACD Indicator')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#Stochastic Oscillator with a Histogram plot for visualization
#Stochastic Oscillator with %K and %D lines
stoch_indicator = ta.momentum.StochasticOscillator(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    window=14,
    smooth_window=3
)

df['stoch_k'] = stoch_indicator.stoch()
df['stoch_d'] = stoch_indicator.stoch_signal()

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot 1: Price
ax1.plot(df.index, df['close'], label='Close Price', linewidth=2, color='black')
ax1.set_ylabel('Price', fontsize=12)
ax1.set_title('Price and Stochastic Oscillator (%K and %D)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Stochastic with both lines
ax2.plot(df.index, df['stoch_k'], label='%K (Fast)', linewidth=2, color='blue')
ax2.plot(df.index, df['stoch_d'], label='%D (Slow/Signal)', linewidth=2, color='red')

# Add reference lines
ax2.axhline(y=80, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label='Overbought (80)')
ax2.axhline(y=20, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7, label='Oversold (20)')
ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Fill zones
ax2.fill_between(df.index, 80, 100, alpha=0.15, color='red', label='Overbought Zone')
ax2.fill_between(df.index, 0, 20, alpha=0.15, color='green', label='Oversold Zone')

ax2.set_ylabel('Stochastic Oscillator (%)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Step 4 --- Label Generation (Target Variable)

#Creating target labels based on future returns
df["future_return"] = df["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1

df["label"] = df.apply(label, axis=1)

def save_processed_csv(df, symbol, interval):

#Save a DataFrame into data/raw/(symbol_interval).csv
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_processed_csv(df, "BTCUSDT", "1dmodified")
