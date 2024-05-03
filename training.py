import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import joblib
import hopsworks


def train_model(stocksymbol, project, ):
        
    # Load dataset
    df = pd.read_csv(stocksymbol+'_df.csv')

    #sanize df names to lovercase and to remove spaces
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop(columns=['dividends', 'stock_splits'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
   
    # Assuming 'Close' is the target variable
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'close']

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

    # Full pipeline: preprocessing and modeling
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor())
    ])
    
    

    # Prepare features and target variable
    X = df[numeric_features]
    
    y = df['close']

    # Perform cross-validation
    scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validated MSE scores: {scores}")
    print(f"Average MSE: {np.mean(scores)}")

    # Fit the model
    model_pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(model_pipeline, stocksymbol+'_linear_regression_model.pkl')
    
    # Define the input schema using the values of X_train
    input_schema = Schema(X.values)

    # Define the output schema using y_train
    output_schema = Schema(y)

    # Create a ModelSchema object specifying the input and output schemas
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # Convert the model schema to a dictionary for further inspection or serialization
    model_schema.to_dict()
    
    mr = project.get_model_registry()

    metrics = {
        "neg_mean_squared_error": np.mean(cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error'))
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
    
    return df

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
    

    


if __name__ == '__main__':
    project = hopsworks.login()
    
    dfs = []
    stocksymbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    for stocksymbol in stocksymbols:
        print(f"Training model for {stocksymbol}")
        dfs.append(train_model(stocksymbol, project))
        upload_to_hopsworks(dfs[-1], project)
    
        
        
        
        
        
    hopsworks.Connection.close(project)
    