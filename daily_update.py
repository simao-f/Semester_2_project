'''
**2nd semester project** - inference pipeline


Elena Skrtic

Simao Ferreira

Laura Keri
'''


import os
from dotenv import load_dotenv
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import polygon 
from datetime import datetime
import time

import tensorflow as tf

import os
from dotenv import load_dotenv

#retrieve the API key from environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

#get the stock data from api
def get_stock_from_api(stocksymbol):
    api_key = POLYGON_API_KEY
    client = polygon.RESTClient(api_key)

    #define the start and end date
    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year - 1)

    #defining which date we will ask from api
    response = client.get_aggs(stocksymbol, 1, 'day', start_date, end_date)
    
    df = pd.DataFrame(response)
    
    #make columns the same, uniform
    df['date'] = df['timestamp']
    df = df['open,high,low,close,volume,date'.split(',')]
    df['symbol'] = stocksymbol
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['volume'] = df['volume'].astype('int64') #because it was a different format
    
    return df
    # Now each column is an attribute (open, high, low, close, volume) and each row is a date.
    
    
#train the model with the new data from the API
def update_model_with_fresh_data(df, stocksymbol): 

    print("updating stock for " + stocksymbol)

    #defining again features and target variable
    #doing the same as in training pipeline
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']
    X = df[numeric_features]
    y = df['close']

    retrieved_model = tf.keras.models.load_model('models/' + stocksymbol + '_lstm_model.h5', compile=False)
    retrieved_model.compile(optimizer='adam', loss='mean_squared_error')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    X_processed = preprocessor.fit_transform(X)

    #creating sequences
    sequence_length = 60 
    X_sequences, y_sequences = [], []
    for i in range(len(X_processed) - sequence_length ):
        X_sequences.append(X_processed[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    #splitting
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    retrieved_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

    #prediction and evaluation
    predictions = retrieved_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    #saving the model
    retrieved_model.save('models/'+stocksymbol + '_lstm_model.h5')

#main function
if __name__ == '__main__':
    print("Weekly update script")
    
    symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    #because we have an api limit of 5 per minute, we need to time it, so we dont get an error
    #so thats 60sec/5=12 +1 for good measure, so we have to wait 13 seconds to call for the api
    start_time = time.time()   
    
    #get stock data for each symbol
    for symbol in symbols:
        df = get_stock_from_api(symbol)
       
        #Check if 13 seconds have passed if not wait until 13 seconds have passed to avoid rate limite of the API 
        end_time = time.time()
        if end_time - start_time < 13:
            waiting_time = 13 - (end_time - start_time)
            
            print(f"Waiting for {waiting_time} seconds")
            time.sleep(waiting_time)
            start_time = time.time()
            print(f"Waiting for {waiting_time} seconds")
            
        # Update the model with the new data    
        update_model_with_fresh_data(df, symbol)
        
        