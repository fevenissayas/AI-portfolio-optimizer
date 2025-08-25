import yfinance as yf
import pandas as pd
import numpy as np

def run_data_ingestion(tickers=['TSLA', 'BND', 'SPY'],
                       start_date='2015-07-01',
                       end_date='2025-07-31',
                       raw_data_path='data/raw/financial_data.csv',
                       processed_data_path='data/processed/adj_close.csv'):
   
    print(f"--- Starting Data Ingestion for {tickers} from {start_date} to {end_date} ---")

    import os
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)
        print('Raw data fetched successfully!')
    except Exception as e:
        print(f'Error fetching raw data: {e}')
        raise

    raw_data.to_csv(raw_data_path)
    print(f'Raw data saved to {raw_data_path}')

    adj_close_data = None
    if isinstance(raw_data.columns, pd.MultiIndex):
        prefs = ['adj close', 'adj', 'adjusted', 'close', 'price']
        def flatten_col(col):
            if isinstance(col, tuple):
                parts = [str(x) for x in col if (x is not None and str(x).strip() != '')]
                return ' '.join(parts).lower()
            return str(col).lower()

        candidates = [col for col in raw_data.columns if any(tok in flatten_col(col) for tok in prefs)]
        if candidates:
            adj_close_data = raw_data[candidates]
            final_cols = []
            for col_tuple in adj_close_data.columns:
                if 'TSLA' in col_tuple:
                    final_cols.append('TSLA')
                elif 'BND' in col_tuple:
                    final_cols.append('BND')
                elif 'SPY' in col_tuple:
                    final_cols.append('SPY')
                else:
                    final_cols.append('_'.join(map(str, col_tuple)).replace('UNNAMED: ', '').replace('_LEVEL_2', ''))
            adj_close_data.columns = final_cols
            adj_close_data = adj_close_data[['TSLA', 'BND', 'SPY']]

        else:
            if 'Adj Close' in raw_data.columns.levels[0]:
                adj_close_data = raw_data['Adj Close']
                adj_close_data.columns = adj_close_data.columns.droplevel(0)
                adj_close_data = adj_close_data[tickers]
            else:
                raise KeyError("Couldn't locate 'Adj Close' or any close-like field in the downloaded data columns for MultiIndex.")
    else:
        cols = list(raw_data.columns)
        prefs = ['adj close', 'adj', 'adjusted', 'close', 'price']
        candidates = [c for c in cols if any(tok in str(c).lower() for tok in prefs)]
        if candidates:
            adj_close_data = raw_data[candidates]
            adj_close_data.columns = [str(c).replace('Adj Close', '').strip() for c in adj_close_data.columns]
            adj_close_data = adj_close_data[tickers]
        else:
            raise KeyError("Couldn't locate 'Adj Close' or any close-like field in the downloaded data columns for single index.")

    if adj_close_data is None:
        raise ValueError("Failed to process adjusted close data.")

    adj_close_data.index = pd.to_datetime(adj_close_data.index)

    adj_close_data.interpolate(method='linear', inplace=True)

    adj_close_data.to_csv(processed_data_path)
    print(f'Processed Adj Close data saved to {processed_data_path}')
    print("Data Ingestion Complete.")

if __name__ == "__main__":
    run_data_ingestion()
