import os
from dotenv import load_dotenv
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
import joblib
import numpy as np
import pandas as pd
import json
import requests
from sklearn.model_selection import cross_val_score
#import api_keys as api_keys
import polygon 
import hopsworks
from datetime import đdatetime
import time


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')


def get_stock_from_api(stocksymbol):
    api_key = POLYGON_API_KEY
    client = polygon.RESTClient(api_key)

    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year - 1)  # Fetching last year's data

    # Fetch aggregates
    response = client.get_aggs(stocksymbol, 7, 'day', start_date, end_date)
    
    df = pd.DataFrame(response)
    
    print(df.columns)
    df['date'] = df['timestamp']
    
    df = df['open,high,low,close,volume,date'.split(',')]
    
    df['symbol'] = stocksymbol
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['volume'] = df['volume'].astype('int64')
    return df

def upload_to_hopsworks(df,project):
    
    fs = project.get_feature_store()
    
    stock_fg = fs.get_or_create_feature_group(
        name="daily_stock_price",
        version=1,
        description="",
        primary_key=["date", "symbol"],
        event_time="date",
    )   
    
    stock_fg.insert(df)
    
def retrieve_model_from_hopsworks(project, stocksymbol):
    mr = project.get_model_registry()
    stock_model = mr.get_model('models/' + stocksymbol + '_lstm_model.h5') #lekérjük a modelt
    return stock_model

def update_model_with_fresh_data(df, symbol):
    stocksymbol = symbol
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']
    
    X = df[numeric_features]
    y = df['close']

    # Load the LSTM model
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
    for i in range(len(X_processed) - sequence_length):
        X_sequences.append(X_processed[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    # Retrain the model
    retrieved_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

    # Evaluate the model
    predictions = retrieved_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the updated model
    retrieved_model.save('models/'+stocksymbol + '_lstm_model.h5')

    # Update model in Hopsworks
    mr = project.get_model_registry()
    
    # Define the input schema using the values of X_train
    input_schema = Schema(X_train)

    # Define the output schema using y_train
    output_schema = Schema(y_train)

    # Create a ModelSchema object specifying the input and output schemas
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # Define the metrics
    metrics = {
        "mean_squared_error": mse
    }

    # Create a new model in the model registry
    stock_model = mr.python.create_model(
        name="lstm_stock_model_" + stocksymbol,     # Name for the model
        metrics=metrics,                      # Metrics used for evaluation
        model_schema=model_schema,            # Schema defining the model's input and output
        input_example=X_sequences[0],       # Example input data for reference
        description="Stock Predictor LSTM",  # Description of the model
    )   
    
    stock_model.save('models/'+stocksymbol + '_lstm_model.h5')

if __name__ == '__main__':
    print("Weekly update script")
    
    symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    project = hopsworks.login()
    start_time = time.time()   
    for symbol in symbols:
        df = get_stock_from_api(symbol)
        upload_to_hopsworks(df, project)
        #stock_model = retrieve_model_from_hopsworks(project, symbol)
        #check if 15 seconds have passed if not wait until 15 seconds have passed
        #to avoid rate limiting of the API
        end_time = time.time()
        if end_time - start_time < 15:
            waiting_time = 15 - (end_time - start_time)
            time.sleep(waiting_time)
            print(f"Waiting for {waiting_time} seconds")
        update_model_with_fresh_data(df, symbol)
    
    hopsworks.Connection.close(project)



'''import os
from dotenv import load_dotenv
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
import joblib
import numpy as np
import pandas as pd
import json
import requests
from sklearn.model_selection import cross_val_score
import api_keys as api_keys
import polygon 
import hopsworks
from datetime import datetime
import time

def get_stock_from_api(stocksymbol):
    api_key = "RH1ov8ypgp3HZyRj6nRVqICs8K4HOdGe"
    client = polygon.RESTClient(api_key)

    end_date = datetime.now().date()
    start_date = end_date.replace(year=end_date.year - 1)  # Fetching last year's data

    # Fetch aggregates
    response = client.get_aggs(stocksymbol, 7, 'day', start_date, end_date)
    

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
    
    

def upload_to_hopsworks(df,project):
    
    fs = project.get_feature_store()
    
    stock_fg = fs.get_or_create_feature_group(
        name="daily_stock_price",
        version=1,
        description="",
        primary_key=["date", "symbol"],
        event_time="date",
    )   
    
    stock_fg.insert(df)
    
    #megnézzük a modelregistryt aztán lekérjük
def retrieve_model_from_hopsworks(project, stocksymbol):
    mr = project.get_model_registry()
    stock_model = mr.get_model("regressor_stock_model_" + stocksymbol) #lekérjük a modelt
    return stock_model

def update_model_with_fresh_data(df, model, symbol):
    stocksymbol = symbol
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']
    
    X = df[numeric_features]
    
    y = df['close']
    retrieved_model = joblib.load(stocksymbol+'_linear_regression_model.pkl')

    retrieved_model.named_steps['regressor'].partial_fit(X, y)
    
    joblib.dump(retrieved_model, stocksymbol+'_linear_regression_model.pkl')
    
    #van model registry = mr
    mr = project.get_model_registry()
    
    # Define the input schema using the values of X_train
    input_schema = Schema(X.values)

    # Define the output schema using y_train
    output_schema = Schema(y)

    # Create a ModelSchema object specifying the input and output schemas
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # Convert the model schema to a dictionary for further inspection or serialization
    #felesleges
    model_schema.to_dict()
    
    #lekérjük a model registryt
    mr = project.get_model_registry()

    metrics = {
        "neg_mean_squared_error": np.mean(cross_val_score(retrieved_model, X, y, cv=5, scoring='neg_mean_squared_error'))
    }

    # Create a new model in the model registry
    stock_model = mr.python.create_model(
        name="regressor_stock_model_" + stocksymbol,     # Name for the model
        metrics=metrics,                      # Metrics used for evaluation
        model_schema=model_schema,            # Schema defining the model's input and output
        input_example=X.sample(),       # Example input data for reference
        description="Stock Predictor",  # Description of the model
    )   
    
    stock_model.save(stocksymbol+'_linear_regression_model.pkl')
    

if __name__ == '__main__':
    print("Daily update script")
    
    symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    project = hopsworks.login()
    start_time = time.time()   
    for symbol in symbols:
        df = get_stock_from_api(symbol)
        upload_to_hopsworks(df, project)
        stock_model = retrieve_model_from_hopsworks(project, symbol)
        #check if 15 seconds have passed if not wait until 15 seconds have passed
        #to avoid rate limiting of the API
        end_time = time.time()
        if end_time - start_time < 15:
            waiting_time = 15 - (end_time - start_time)
            time.sleep(waiting_time)
            print(f"Waiting for {waiting_time} seconds")
        update_model_with_fresh_data(df, stock_model, symbol)
        
    hopsworks.Connection.close(project)
    '''