import os
from dotenv import load_dotenv
import pandas as pd
import json
import requests
import api_keys as api_keys

def get_stock_from_api(stocksymbol):

    load_dotenv()

    url = (
        f'https://www.alphavantage.co/query?'
        f'function=TIME_SERIES_DAILY&'
        f'symbol={stocksymbol}&'
        f'apikey={api_keys.api_key}'
    )

    response = requests.get(url)

    data = response.json()

    # Convert the list of articles into a DataFrame
    finance = data['Time Series (Daily)']
    symbol = data['Meta Data']['2. Symbol']

    df = pd.DataFrame(finance)

    # Transpose the DataFrame
    df_transposed = df.transpose()
    df_transposed['Symbol'] = symbol
    return df, df_transposed, data
    # Now, each column is an attribute (open, high, low, close, volume) and each row is a date.
    