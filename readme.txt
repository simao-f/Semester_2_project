
Serverless Stock Market Prediction Application

This project is based on a serverless design which means that the pipelines are implemented separately to enhance the robustness of the application.
In the interface the user will be able to select the ticker symbol for a given stock in the list. Based on this, the app will give an up-to-date historical data on the closing price of the stock and also show a prediction for the price of the same stock symbol. 

(Before starting this project we created the EDA.ipynb file to understand the data)


This project consists of four main components:

1. Feature pipeline (file: feature_pipeline.ipynb)

This pipeline takes historic stock data from yfinance and save is as csv.
This raw data is then processed to contain desired features (feature engineering) in the correct data format in order to be interpreted by the training pipeline. 


2. Training pipeline (files: training.py)

This pipeline takes the files from the previous pipeline and uses that data to train the model using a linear regression model.
Using the feature store, the model will be able to quickly access the data.
From there, the model is trained as usual, after which it is deployed on Hopsworks for easy access when doing inference.


3. Inference pipeline (file: daily_update.py)

Using the trained model accessible on Hopsworks, this pipeline automatically updates stock trading models with fresh data from the Polygon API we're using in our project. This fresh data is used to update our model 


4. User interface (file: interface2.py)

Finally we created an accessible UI is provided as a web application developed using Streamlit. 
This application allows the user to select any stock from the list.
From here, the application will display historical closing price data of the selected stock and a prediction of the price variation.



Requirements for running the code

Download the repo, create a new conda environment and then run:

pip install -r requirements.txt
conda environment (python 3.11.9)