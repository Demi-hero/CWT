import pandas as pd
import requests
import json
import warnings

def sentiment_analysis(input_data):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if isinstance(input_data, pd.core.series.Series):
        payload = {'data': json.dumps(input_data.tolist())}
    elif isinstance(input_data, list) | isinstance(input_data, str):
        payload = {'data': json.dumps(input_data)}
    else:
        warnings.warn("This function only takes pd.Series, lists, or strings as input", UserWarning)
        return -1
    return requests.post('http://127.0.0.1:5000/classifier', headers=headers, json=payload).json()