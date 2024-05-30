import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import hopsworks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train an LSTM model. stocksymbol and project are the input parameters.
def train_model(stocksymbol, project):
        
    # Load dataset
    df = pd.read_csv(stocksymbol+'_df.csv')

    # Sanitize df columns to lowercase and to remove spaces
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop(columns=['dividends', 'stock_splits'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
   
    # Define numeric features
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']

    # Add stock symbol to the dataframe, so we can store all in one feature group
    df['symbol'] = stocksymbol
    
    # Create pipelines for numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
        ('scaler', StandardScaler())  # Scale features
    ])
    print(numeric_features)

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    # Prepare features and target variable
    X = df[numeric_features]
    print(X.head())
    y = df['close']
    
    # Preprocess the data
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

    # Create and train the LSTM model
    lstm_model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

    # Evaluate the model
    predictions = lstm_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the trained model
    lstm_model.save('models/'+stocksymbol + '_lstm_model.h5')

    # Define the input schema using the values of X_train
    input_schema = Schema(X_train.reshape(-1, X_train.shape[2]))

    # Define the output schema using y_train
    output_schema = Schema(y_train)

    # Create a ModelSchema object specifying the input and output schemas
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # Convert the model schema to a dictionary for further inspection or serialization
    model_schema.to_dict()
    
    # Get the model registry
    mr = project.get_model_registry()

    # Define the metrics for the model
    metrics = {
        "mean_squared_error": mse
    }

    # Create a new model in the model registry
    stock_model = mr.python.create_model(
        name="lstm_stock_model_" + stocksymbol,     # Name for the model
        metrics=metrics,                      # Metrics used for evaluation
        model_schema=model_schema,            # Schema defining the model's input and output
        input_example=X_sequences[0],         # Example input data for reference
        description="Stock Predictor LSTM",  # Description of the model
    )   
    
    stock_model.save('models/'+stocksymbol + '_lstm_model.h5')
    
    return df

# Upload the data to Hopsworks feature store
def upload_to_hopsworks(df, project):
    
    fs = project.get_feature_store()
    
    stock_fg = fs.get_or_create_feature_group(
        name="daily_stock_price",
        version=1,
        description="",
        primary_key=["date", "symbol"],
        event_time="date",
    )   
    
    stock_fg.insert(df)
    

# Login to Hopsworks, upload the trained model and the data to the feature store 
# This is the main function. We login to hopsworks,  
if __name__ == '__main__':
    print("im alive")
    project = hopsworks.login()
    
    dfs = []
    stocksymbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    # Train models for each stock symbol
    for stocksymbol in stocksymbols:
        print(f"Training model for {stocksymbol}")
        dfs.append(train_model(stocksymbol, project)) # what's appended here is the dataframe
        upload_to_hopsworks(dfs[-1], project)
    
    # Close the connection to Hopsworks
    hopsworks.Connection.close(project)
    print("im dead")