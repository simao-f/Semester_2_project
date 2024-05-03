import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import joblib

def train_model(stocksymbol):
        
    # Load dataset
    df = pd.read_csv(stocksymbol+'_df.csv')

    # Assuming 'Close' is the target variable
    numeric_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'Close']

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

    # Full pipeline: preprocessing and modeling
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Prepare features and target variable
    X = df[numeric_features]
    y = df['Close']

    # Perform cross-validation
    scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validated MSE scores: {scores}")
    print(f"Average MSE: {np.mean(scores)}")

    # Fit the model
    model_pipeline.fit(X, y)

    # Save the trained model
    joblib.dump(model_pipeline, stocksymbol+'_linear_regression_model.pkl')


if __name__ == '__main__':
    stocksymbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA']
    for stocksymbol in stocksymbols:
        print(f"Training model for {stocksymbol}")
        train_model(stocksymbol)