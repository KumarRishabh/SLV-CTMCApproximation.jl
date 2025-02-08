import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

def get_expiration_dates(symbol: str) -> List[str]:
    """Get all available expiration dates for options"""
    ticker = yf.Ticker(symbol)
    return ticker.options

def get_options_chain(symbol: str, expiration_date: str) -> Optional[Dict]:
    """Fetch options chain data for given date"""
    ticker = yf.Ticker(symbol)
    try:
        options = ticker.option_chain(expiration_date)
        return {
            'calls': options.calls,
            'puts': options.puts
        }
    except Exception as e:
        print(f"Error fetching data for {expiration_date}: {e}")
        return None

def process_options_data(options_data: Dict, expiration_date: str) -> pd.DataFrame:
    """Process options data and add computed columns"""
    calls = options_data['calls']
    calls['expirationDate'] = expiration_date
    calls['timeToExpiry'] = (pd.to_datetime(expiration_date) - pd.Timestamp.now()).days / 365.0
    return calls[['strike', 'lastPrice', 'impliedVolatility', 'expirationDate', 'timeToExpiry']]

def main():
    symbol = 'NVDA'
    all_calls = pd.DataFrame()
    
    # Fetch and process data
    exp_dates = get_expiration_dates(symbol)
    print(f"Found {len(exp_dates)} expiration dates")
    
    for date in exp_dates:
        options_data = get_options_chain(symbol, date)
        if options_data:
            processed_data = process_options_data(options_data, date)
            all_calls = pd.concat([all_calls, processed_data])
    
    # Save processed data
    output_file = f'{symbol}_options_data.csv'
    all_calls.to_csv(output_file, index=False)
    print(f"Saved {len(all_calls)} options records to {output_file}")

if __name__ == "__main__":
    main()