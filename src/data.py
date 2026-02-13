import time
import yfinance as yf
import pandas as pd


def download_one(ticker, start="2020-01-01", end=None, retries=3, sleep_sec=2):
    last_err = None
    for i in range(retries):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,   # evita concorrência no Windows
            )
            if data is None or data.empty:
                raise RuntimeError(f"Empty data for {ticker}")
            close = data[["Close"]].rename(columns={"Close": ticker})
            return close
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"Failed to download {ticker} after {retries} tries. Last error: {last_err}")


def download_prices(tickers, start="2020-01-01", end=None):
    frames = []
    for t in tickers:
        print(f"Downloading {t}...")
        try:
            frames.append(download_one(t, start=start, end=end))
        except Exception as e:
            print(f"WARNING: {t} failed: {e}")
    if not frames:
        raise RuntimeError("No tickers downloaded successfully.")
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.dropna(how="all")
    return prices


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "TLT"]
    prices = download_prices(tickers, start="2020-01-01")
    print(prices.tail())
