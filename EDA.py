import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from utils import txt_processing as txt
from wordcloud import WordCloud
# nltk.download('stopwords')
# nltk.download('punkt')
stop_word_builds = 0

if __name__ == "__main__":

    # Initial Data cleaning to remove all the unneeded information from the reviews
    stop = stopwords.words('english') + txt.extra_stops
    # Add in the intersection of the top 50 positive and negative review words
    data = pd.read_csv("Datasets/London_reviews.csv")
    data = data.iloc[:, 2:]
    data['sentiment'] = data['sample'].map({"Positive": 1, "Negative": 0})
    data = data.loc[~pd.isna(data['review_full']), ]
    # data = data[['review_full', 'rating_review']]
    data['rating_review'] = data['rating_review'].astype(float).astype(int)
    data = txt.clean_text(data, target_column='review_full', stop_words=stop)
    # data = pd.read_csv("Datasets\\Cleaned_reviews.csv")
    # Building out the additional stop words where there is large overlap between reviews.
    # Intent is to force the model to learn from less common features amongst the reviews
    # In theory this process can be run multiple times
    for reps in range(stop_word_builds):
        new_stops = txt.stop_word_builder(data, stop)
        stop += new_stops
        stop = list(set(stop))
    data = txt.clean_text(data, target_column='review_full', stop_words=stop)
    # data.to_csv("Datasets\\Cleaned_reviews.csv")
    # data = pd.read_csv("Datasets\\Cleaned_reviews.csv")
    # Average Review score
    agg_avg = data.rating_review.mean()
    agg_median = data.rating_review.median()
    # Average Review Length in characters
    data['character_count'] = [len(x) for x in data['review_full']]
    review_character_quartiles = np.quantile(data['character_count'], [0, .33, .66, 1])
    data['character_count_group'] = pd.cut(data['character_count'], review_character_quartiles, labels=['low', 'med', 'high'])
    data.groupby("character_count_group").agg(review_avg=("rating_review", "mean"))
    # Average Words per review
    data['word_count'] = [len(re.findall(r'\w+', x)) for x in data['review_full']]
    # Relationship with Review Score?
    review_word_quartiles = np.quantile(data['word_count'], [0, .33, .66, 1])
    data['word_count_group'] = pd.cut(data['word_count'], review_word_quartiles, labels=['low', 'med', 'high'])
    data.groupby("word_count_group").agg(review_avg=("rating_review", "mean"))

    # For the top and bottom n rated restaurants what in general were they serving?
    avg_ratings = data.groupby("restaurant_name").agg(review_avg=("rating_review", "mean"),
                                                      reviews=("rating_review", "count")).\
        sort_values(by=['review_avg',  'reviews'], ascending=False)

    # High Trust Per review
    top_ten_ht = avg_ratings.loc[avg_ratings['reviews'] > 100, :].head(10)
    bottom_ten_ht = avg_ratings.loc[avg_ratings['reviews'] > 100, :].tail(10)

    # Low Trust Per review
    top_ten_lt = avg_ratings.loc[avg_ratings['reviews'] > 500, :].head(10)
    top_ten = top_ten_lt.index.to_list()
    top_ten = [('top', x) for x in top_ten]

    bottom_ten_lt = avg_ratings.loc[avg_ratings['reviews'] > 500, :].tail(10)
    bottom_ten = bottom_ten_lt.index.to_list()
    bottom_ten = [('bottom', x) for x in bottom_ten]
    rests_breakdown = {}
    food = wn.synset('food.n.02')
    dishes = wn.synset('dish.n.02')
    service = wn.synset('service.n.02')
    food = list(set([w for s in food.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    food += list(set([w for s in dishes.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    for food_place in bottom_ten+top_ten:
        test_data = data.loc[data['restaurant_name'] == food_place[1], :]
        flat_list = [item for sublist in test_data['lem_reviews'] for item in sublist]
        food_list = [item for item in flat_list if item in food]
        nlp_words = nltk.FreqDist(food_list)
        wcloud = WordCloud().generate_from_frequencies(nlp_words)
        plt.imshow(wcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"{food_place[0]}_{food_place[1]}_Wordcloud.png")
        plt.clf()

    # For Reviewers with <3, 3-10 and 10+ same questions
    # what was their average review legnth.
    audience_grouping = data.groupby('author_id').agg(reviews=("rating_review", "count"),
                             avg_word_count=("word_count", "mean"),
                             avg_review_score=("rating_review", "mean"))
    audience_review_len_groupings = np.quantile(audience_grouping['avg_word_count'], [0, .33, .66, 1])
    audience_grouping['word_count_group'] = pd.cut(audience_grouping['avg_word_count'],
                                                   audience_review_len_groupings, labels=['low', 'med', 'high'])
    audience_grouping['review_count_group'] = pd.cut(audience_grouping['reviews'],
                                                   [0, 3, 10, max(audience_grouping['avg_word_count'])],
                                                     labels=['0-3', '3-10', '10+'])
    audience_grouping.groupby('word_count_group').agg(reviews=("reviews", "count"),
                                                      avg_review_score=("avg_review_score", "mean"),
                             avg_word_count=("avg_word_count", "mean"))

    audience_grouping.groupby('review_count_group').agg(reviews=("reviews", "count"),
                                                      avg_review_score=("avg_review_score", "mean"),
                                                      avg_word_count=("avg_word_count", "mean"))
    # what was their most common words?

