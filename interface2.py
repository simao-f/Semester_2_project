import streamlit as st
import hopsworks
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
from hsfs.feature import Feature

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Hopsworks connection
project = hopsworks.login()
feature_store = project.get_feature_store()

# Fetch available stock symbols from the feature store
symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA'] 
stock_symbol = st.selectbox("Select stock symbol", symbols)
days_range = st.slider("Select days range", min_value=7, max_value=90, step=1)

# Load historical data for the selected symbol
feature_group = feature_store.get_feature_group(name="daily_stock_price", version=1)
feature_group = feature_group.select(["date", "high", "low", "close", "open", "volume"]).filter(Feature("symbol").like(stock_symbol))

# Load and plot historical price data for stock symbol from Hopsworks feature store
df = feature_group.read()

# Sort data by date
feature_df = df.sort_values("date", ascending=False)

# Plot the actual data
fig_actual = px.line(
    feature_df,
    x='date',
    y=['high', 'low', 'close'],
    title=f"Actual Data for {stock_symbol}",
    labels={'value': 'Price', 'date': 'Date'}
)
st.plotly_chart(fig_actual)


# Prepare data for prediction (using linear regression as an example)
numeric_features = [col for col in feature_df.columns if feature_df[col].dtype in ['int64', 'float64'] and col != 'close']
X = feature_df[numeric_features]


# x-eket sequencekke kell alakitani elotte aaaaaaaaaaaaaa
# Load the trained linear regression model, predict future prices, and plot the predictions    
retrieved_model = tf.keras.models.load_model('models/' + stock_symbol + '_lstm_model.h5', compile=False)

# Predict future prices

sequence_length = 60  # Example sequence length
X_sequences = []
for i in range(len(X) - sequence_length):
    X_sequences.append(X[i:i + sequence_length])

X_sequences = np.array(X_sequences)

predicted_close = retrieved_model.predict(X_sequences)
print(len(predicted_close))
print(len(feature_df))

feature_df = feature_df.iloc[60:]

# Plot predicted data
feature_df['predicted_close'] = predicted_close
print(feature_df.head)

fig_predicted = px.line(
    feature_df,
    x='date',
    y='predicted_close',
    title=f"Predicted Close Price for {stock_symbol}",
    labels={'predicted_close': 'Predicted Close', 'date': 'Date'}
)
st.plotly_chart(fig_predicted)