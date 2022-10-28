import pandas as pd
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings

def call_sentiment_api(input_data):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if isinstance(input_data, pd.core.series.Series):
        payload = {'data': json.dumps(input_data.tolist())}
    elif isinstance(input_data, list) | isinstance(input_data, str):
        payload = {'data': json.dumps(input_data)}
    else:
        warnings.warn("This function only takes pd.Series, lists, or strings as input", UserWarning)
        return -1
    return requests.post('http://127.0.0.1:5000/classifier', headers=headers, json=payload).json()

data = pd.read_csv("Datasets/London_reviews.csv")
data = data[['review_full', 'rating_review']]
data = data.loc[~pd.isna(data['review_full']), ]
data['rating_review'] = data['rating_review'].astype(float).astype(int)
data_sample = data.sample(frac=0.01)
# data_sample.to_csv("datasample.csv", index=False)
data_sample =pd.read_csv("datasample.csv")
X_train, X_test, y_train, y_test = \
    train_test_split(data_sample['review_full'], data_sample['rating_review'], random_state=12, test_size=0.1)
test_predictions = call_sentiment_api(X_test)