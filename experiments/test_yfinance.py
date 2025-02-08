import yfinance as yf

def test_yfinance():
    msft = yf.Ticker("MSFT")
    print(f"Current price: {msft.info['regularMarketPrice']}")

if __name__ == "__main__":
    test_yfinance()