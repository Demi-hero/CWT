import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import call_api



if __name__ == '__main__':

    if not os.path.isfile('datasample.csv'):
        data = pd.read_csv("Datasets/London_reviews.csv")
        data = data[['review_full', 'rating_review']]
        data = data.loc[~pd.isna(data['review_full']), ]
        data['rating_review'] = data['rating_review'].astype(float).astype(int)
        data_sample = data.sample(frac=0.01)
        data_sample.to_csv("datasample.csv", index=False)
    else:
        data_sample =pd.read_csv("datasample.csv")
    X_train, X_test, y_train, y_test = \
        train_test_split(data_sample['review_full'], data_sample['rating_review'], random_state=12, test_size=0.1)
    predictions = call_api.sentiment_analysis(X_test)
    print(predictions)
