import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# nltk.download('punkt')
words = stopwords.words("english")
vectorizer = TfidfVectorizer(min_df=3, stop_words="english", sublinear_tf=True)

data = pd.read_csv("Datasets\\Cleaned_reviews.csv")
X = data.clean_reviews
X2 = data.lem_reviews
y = data.rating_review-1
del data
text = " ".join(X.to_list())
words: list[str] = nltk.word_tokenize(text)
fd = nltk.FreqDist(words)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', RandomForestClassifier())])

model = pipeline.fit(X_train, y_train)
with open('randforest_test.pickle', 'wb') as f:
    pickle.dump(model, f)

ytest = np.array(y_test)

print(confusion_matrix(ytest, model.predict(X_test)))
print(classification_report(ytest, model.predict(X_test)))

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

target_names = ['0', '1', '2', '3', '4']
print("top 10 keywords per class:")
for i, label in enumerate(target_names):
    top10 = np.argsort(clf.feature_importances_)[-10:]
    print("%s: %s" % (label, " ".join(feature_names[top10])))