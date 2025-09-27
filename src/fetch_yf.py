# src/fetch_yf.py
import yfinance as yf
import pandas as pd
import uuid
from pathlib import Path
import argparse

def fetch_yahoo_news(ticker, out_csv="./data/yf_news.csv", max_items=20):
    stock = yf.Ticker(ticker)
    news = stock.news[:max_items]  # Yahoo Finance provides recent news articles

    records = []
    for n in news:
        # Each news item is a dict with 'title', 'link', 'publisher', 'providerPublishTime'
        doc_id = str(uuid.uuid4())
        text = n.get("title", "")
        date = n.get("providerPublishTime")
        source = n.get("publisher")
        records.append({
            "id": doc_id,
            "text": text,
            "date": date,
            "source": source,
            "ticker": ticker
        })

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} news items for {ticker} to {out_csv}")
    return out_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL")
    parser.add_argument("--out", default="./data/yf_news.csv")
    args = parser.parse_args()
    fetch_yahoo_news(args.ticker, args.out)
