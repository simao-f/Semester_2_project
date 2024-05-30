'''
**2nd semester project** - interface


Elena Skrtic

Simao Ferreira

Laura Keri
'''


import streamlit as st
import hopsworks
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
from hsfs.feature import Feature

import tensorflow as tf

#connecting to hopsworks
project = hopsworks.login()
feature_store = project.get_feature_store()

#fetch available stock symbols from the feature store
symbols = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'MSFT', 'META', 'NVDA'] 
stock_symbol = st.selectbox("Select stock symbol", symbols)
days_range = st.slider("Select days range", min_value=7, max_value=90, step=1)

#load historical data for the selected symbol
feature_group = feature_store.get_feature_group(name="daily_stock_price", version=1)
feature_group = feature_group.select(["date", "high", "low", "close", "open", "volume"]).filter(Feature("symbol").like(stock_symbol))

#load and plot historical price data for stock symbol from Hopsworks feature store
df = feature_group.read()

#sort data by date
feature_df = df.sort_values("date", ascending=False)

#plot for the actual stock prices - so we can compare them to the predicted ones
fig_actual = px.line(
    feature_df,
    x='date',
    y=['high', 'low', 'close'],
    title=f"Actual Data for {stock_symbol}",
    labels={'value': 'Price', 'date': 'Date'}
)
st.plotly_chart(fig_actual)

#prepare data and load in model for prediction
numeric_features = [col for col in feature_df.columns if feature_df[col].dtype in ['int64', 'float64'] and col != 'close']
X = feature_df[numeric_features]
retrieved_model = tf.keras.models.load_model('models/' + stock_symbol + '_lstm_model.h5', compile=False)

#needs sequences again, so LSTM model works
#and then makin gthe prediction
sequence_length = 60 
X_sequences = []
for i in range(len(X) - sequence_length):
    X_sequences.append(X[i:i + sequence_length])

X_sequences = np.array(X_sequences)

predicted_close = retrieved_model.predict(X_sequences)
print(len(predicted_close))
print(len(feature_df))

feature_df = feature_df.iloc[60:]

#now plotting the predicted data
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