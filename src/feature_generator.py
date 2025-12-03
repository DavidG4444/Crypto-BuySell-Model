#Importing Libraries
import talib as ta
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
upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)

df['bb_upper'] = upper
df['bb_middle'] = middle
df['bb_lower'] = lower

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

#MACD with a Histogram plot
macd, signal, histogram = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

df['macd'] = macd
df['macd_signal'] = signal
df['macd_histogram'] = histogram

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

#Stochastic Oscillator
