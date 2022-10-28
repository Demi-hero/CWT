import pandas as pd
import pickle
from sqlite3 import Error
import sqlite3
# nltk.download('wordnet')

# print(df.head())
data = pd.read_csv("Datasets\\Cleaned_reviews.csv")
X2 = data['lem_reviews']
y = data['rating_review']-1

pickle_load = open('randforest_test.pickle', 'rb')
clf2 = pickle.load(pickle_load)

def classify(model, strings):
    list = []
    for bit in strings:
        bit = str(bit)
        texty = [bit]
        pred = model.predict(texty)
        list.append(pred[0])
    return list


X_test
y_test
clf2.predict(X_test)
predictions = classify(clf2, X_test)
predictions
y_test
data['classed'] = predictions