import os
import requests
#print(requests.__version__)
import pandas as pd

def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","num_trades",
                  "taker_base_volume","taker_quote_volume","ignore"]
    return df

df = fetch_binance("BTCUSDT", "1d", 1000)
print(df.head())
print(df.info())

def save_raw_csv(df, symbol, interval):

#Save a DataFrame into data/raw/(symbol_interval).csv
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_raw_csv(df, "BTCUSDT", "1d")
