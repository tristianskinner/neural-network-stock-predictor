import yfinance as yf
import os

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
START = "2016-01-01"
END = "2024-12-31"

os.makedirs("data", exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END)
    df.to_csv(f"data/{ticker}.csv")
    print(f"Saved to data/{ticker}.csv")

print("All downloads complete.")
