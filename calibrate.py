import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
import nltk
from utils import txt_processing as txt
from utils import model_processing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
# nltk.download('punkt')

if __name__ == '__main__':

    data = pd.read_csv("Datasets\\Cleaned_reviews.csv")
    data = data.loc[~ pd.isna(data['clean_reviews']), :]
    X = data.clean_reviews
    X2 = data.lem_reviews
    y = data.rating_review-1
    del data

    out_path = mp.generate_new_model_directory()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    words = stopwords.words("english") + txt.extra_stops
    vectorizer = TfidfVectorizer(min_df=3, stop_words=words, sublinear_tf=True)

    parameters = {
        'rf': {'chi__k': [1000, 1500, 2000],
                'clf__n_estimators': [100, 500],
                'clf__max_features':  ['sqrt', 'log2'],
                'clf__max_depth': [4, 5, 6],
                'clf__criterion': ['gini', 'entropy']
                },
        'nb': {'vect__min_df':  [1,3,5],
                'vect__max_df': [1.0],
                'vect__max_features': [10000, 15000],
                'vect__ngram_range': [(1,2),(1, 2)],
                'chi__k': [1000, 1500, 2000]}
    }

    pipelines ={'rf': Pipeline([('vect', vectorizer),
                                ('chi',  SelectKBest(chi2)),
                                ('clf', RandomForestClassifier())]),
                'nb': Pipeline([('vect', vectorizer),
                                ('chi', SelectKBest(chi2)),
                                ('clf', MultinomialNB())])
                }

    models = {}
    report = {}
    for model_type in ['nb', 'rf']:
        gs = GridSearchCV(pipelines[model_type], refit=True, param_grid=parameters[model_type], verbose=1)
        gs.fit(X_train, y_train)
        models[model_type] = gs.best_estimator_
        report[model_type] = classification_report(y_test, models[model_type].predict(X_test), output_dict=True)

    mp.save_training_data(X_train, X_test, y_train, y_test, out_path)
    if report['nb']['accuracy'] > report['rf']['accuracy']:
        mp.save_model_data(models['nb'], report['nb'], out_path)
    else:
        mp.save_model_data(models['rf'], report['rf'], out_path)

