'''
**2nd semester project** - training pipeline


Elena Skrtic

Simao Ferreira

Laura Keri
'''

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import hopsworks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#function defining LSTM model that we are about to us
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

#function for training the LSTM model
def train_model(stocksymbol, project):
        
    df = pd.read_csv(stocksymbol+'_df.csv')

    #sanitize df columns to lowercase and to remove spaces to make everything uniform
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop(columns=['dividends', 'stock_splits'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
   
    #numeric features - input features
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']

    #add stock symbol to the dataframe, so we can store all in one feature group
    df['symbol'] = stocksymbol
    
    #creating pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  #filling up missing values with median
        ('scaler', StandardScaler())  #scaling the features
    ])
    print(numeric_features)

    #combine the preprocessing steps for numeric features into a single ColumnTransformer
    #this ensures that the numeric features are both imputed (missing values filled) and scaled 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
        ])

    #features and target variable - our target variable is the *closing* price
    X = df[numeric_features]
    print(X.head())
    y = df['close']
    
    X_processed = preprocessor.fit_transform(X)
    
    #create sequences for LSTM - because it requires sequential data
    #the sequence lenght is just an example
    sequence_length = 60
    X_sequences, y_sequences = [], []
    for i in range(len(X_processed) - sequence_length):
        X_sequences.append(X_processed[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

    #create and train the LSTM model
    lstm_model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

    #evaluating the model
    #we used several metrics
    predictions = lstm_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    y_test_classes = np.where(y_test > y_test.mean(), 1, 0)
    predictions_classes = np.where(predictions > predictions.mean(), 1, 0)
    accuracy = accuracy_score(y_test_classes, predictions_classes)
    f1 = f1_score(y_test_classes, predictions_classes, average='weighted')
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    #save the trained model into a .h5 file
    lstm_model.save('models/'+stocksymbol + '_lstm_model.h5')

    # Define and serialize the input and output schemas for the model using training data
    input_schema = Schema(X_train.reshape(-1, X_train.shape[2]))
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    model_schema.to_dict()
    
    #get the model registry, so we can upload it to hopsworks
    mr = project.get_model_registry()

    #define the metrics for the model that we will use for evaluation
    metrics = {
        "mean_squared_error": mse
    }

    #Create a new model in the model registry
    stock_model = mr.python.create_model(
        name="lstm_stock_model_" + stocksymbol,     #name for the model
        metrics=metrics,                      #previously defined model metrics
        model_schema=model_schema,            #model's input and output
        input_example=X_sequences[0],         #example input data for reference
        description="Stock Predictor LSTM",  #description of the model
    )   
    
    #saving the model
    stock_model.save('models/'+stocksymbol + '_lstm_model.h5')
    
    return df

#upload the data to Hopsworks feature store
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
    

#login to Hopsworks, upload the trained model and the data to the feature store 
#this is the main function. We login to hopsworks,  
if __name__ == '__main__':
    print("starting training")
    project = hopsworks.login()
    
    dfs = []
    stocksymbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    
    #rain models for each stock symtbol
    for stocksymbol in stocksymbols:
        print(f"Training model for {stocksymbol}")
        dfs.append(train_model(stocksymbol, project)) # what's appended here is the dataframe
        upload_to_hopsworks(dfs[-1], project)
    
    #close the connection to Hopsworks
    hopsworks.Connection.close(project)
    print("training finished")