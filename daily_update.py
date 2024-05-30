import os
from dotenv import load_dotenv
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
import joblib
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
#ez mükszik

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


import os
from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv("api.env")

# Retrieve the API key from environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

# Get the stock data from the API
def get_stock_from_api(stocksymbol):
    api_key = POLYGON_API_KEY
    client = polygon.RESTClient(api_key)

    #lekérjük a mostani időt, to end data, from start date 
    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year - 1)  # Fetching last year's data

    #a stockszimbolt le kell kérni napi összesítésbe + megadjuk, hogy mettől meddig
    # Fetch aggregates
    response = client.get_aggs(stocksymbol, 1, 'day', start_date, end_date)
    

    df = pd.DataFrame(response)
    
    
    #ezt csak azért, hogy ugyanaz legyen, mint trainingbe (ez a transform része)
    # Rename the columns
    df['date'] = df['timestamp']
    
    # Drop the timestamp column
    df = df['open,high,low,close,volume,date'.split(',')]

    # Add the stock symbol to the dataframe
    df['symbol'] = stocksymbol
    
    # Convert the 'date' column to datetime format, setting timezone to UTC
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    #előtte float volt es a trainingbe is int volt
    # Change the data type of the 'volume' column to integer for consistency and computation
    df['volume'] = df['volume'].astype('int64')
    
    return df
    # Now each column is an attribute (open, high, low, close, volume) and each row is a date.
    
    
# Train the model with the new data from the API
def update_model_with_fresh_data(df, stocksymbol): 
    # stocksymbol = df['symbol'][0] #helyette: (inputként lehetett volna a stocksymbol és akkor nem kéne ez a sor)
    
    # Convert the list of articles into a DataFrame
    
    print("updating stock for " + stocksymbol)
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']
    
    X = df[numeric_features]
    y = df['close']
    # Load the model
    #betöltjük a lementett modelt
    
    retrieved_model = tf.keras.models.load_model('models/' + stocksymbol + '_lstm_model.h5', compile=False)
    retrieved_model.compile(optimizer='adam', loss='mean_squared_error')

    # Create pipelines for numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
        ('scaler', StandardScaler())  # Scale features
    ])

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    X_processed = preprocessor.fit_transform(X)

    # Create sequences for LSTM
    sequence_length = 60  # Example sequence length
    X_sequences, y_sequences = [], []
    for i in range(len(X_processed) - sequence_length ):
        X_sequences.append(X_processed[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    retrieved_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

    # Evaluate the model
    predictions = retrieved_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the updated model
    retrieved_model.save('models/'+stocksymbol + '_lstm_model.h5')

if __name__ == '__main__':
    print("Weekly update script")
    
    symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    #kijátsszuk az APInak a limitjét, mert x kérést enged perceknént 60mp/5=12 plusz 1, aztán várunk, hogy biztos elteljen a 13mp, mert amugy meg hibás lenne
    start_time = time.time()   
    
    # Get stock data for each symbol
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
        
        