import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer # Equivalent to CountVectorizer followed by TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from utils import assement as asm
from utils import txt_processing as txt




if __name__ ==  "__main__":

    # Initial Data cleaning to remove all the unneeded information from the reviews
    stop = stopwords.words('english')
    # Add in the intersection of the top 50 positive and negative review words
    data = pd.read_csv("Datasets/London_reviews.csv")
    data = data.iloc[:, 2:]
    data['sentiment'] = data['sample'].map({"Positive": 1, "Negative":0})
    data = data.loc[~pd.isna(data['review_full']), ]
    data = data[['review_full', 'rating_review']]
    data['rating_review'] = data['rating_review'].astype(float).astype(int)
    data = txt.clean_text(data, target_column='review_full', stop_words=stop)

    data.to_csv("Datasets\\Cleaned_reviews.csv")
    data = pd.read_csv("Datasets\\Cleaned_reviews.csv")


    