import os
from dotenv import load_dotenv
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import polygon 
from datetime import datetime
import time

def get_stock_from_api(stocksymbol):
    api_key = "RH1ov8ypgp3HZyRj6nRVqICs8K4HOdGe"
    client = polygon.RESTClient(api_key)

    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year - 1)  # Fetching last year's data

    # Fetch aggregates
    response = client.get_aggs(stocksymbol, 1, 'day', start_date, end_date)
    

    df = pd.DataFrame(response)
    
    print(df.columns)
    df['date'] = df['timestamp']
    
    
    df = df['open,high,low,close,volume,date'.split(',')]
    # Convert the list of articles into a DataFrame
    

    df['symbol'] = stocksymbol
    
    #df_transposed['date'] = pd.to_datetime(df_transposed['date'], utc=True)
    # Assuming 'Close' is the target variable
    
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    # Assuming 'Close' is the target variable
    df['volume'] = df['volume'].astype('int64')
    return df
    # Now, each column is an attribute (open, high, low, close, volume) and each row is a date.
    
    

def update_model_with_fresh_data(df):
    stocksymbol = df['symbol'][0]
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']
    
    X = df[numeric_features]
    
    y = df['close']
    retrieved_model = joblib.load(stocksymbol+'_linear_regression_model.pkl')

    retrieved_model.named_steps['regressor'].partial_fit(X, y)
    
    joblib.dump(retrieved_model, stocksymbol+'_linear_regression_model.pkl')
    

    

if __name__ == '__main__':
    print("Daily update script")
    
    symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    start_time = time.time()   
    for symbol in symbols:
        df = get_stock_from_api(symbol)
        #check if 15 seconds have passed if not wait until 15 seconds have passed
        #to avoid rate limiting of the API
        end_time = time.time()
        if end_time - start_time < 15:
            waiting_time = 15 - (end_time - start_time)
            time.sleep(waiting_time)
            print(f"Waiting for {waiting_time} seconds")
        update_model_with_fresh_data(df)
        
        