import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from utils import txt_processing as txt
from wordcloud import WordCloud
import matplotlib.style as style
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
    rating_dist = data.groupby('rating_review').agg(reviews=('Unnamed: 0',"count")).reset_index()

    # Score Dist plot
    sns.set(rc={'figure.figsize': (15, 8.27)})
    bplot = sns.barplot(x='rating_review', y='reviews', data=rating_dist)
    sns.set_context("poster")
    fig =bplot.get_figure()
    fig.savefig('score_dist.png')
    plt.clf()

    # Average Words per review
    data['word_count'] = [len(re.findall(r'\w+', x)) for x in data['review_full']]
    # Relationship with Review Score?
    review_word_quartiles = np.quantile(data['word_count'], [0, .33, .66, 1])
    data['word_count_group'] = pd.cut(data['word_count'], review_word_quartiles, labels=['low', 'med', 'high'])
    word_count_dist = data.groupby("word_count_group").agg(review_avg=("rating_review", "mean")).reset_index()

    # word dist plot
    sns.set(rc={'figure.figsize': (15, 8.27)})
    bplot = sns.barplot(x='word_count_group', y='review_avg', data=word_count_dist)
    sns.set_context("poster")
    fig = bplot.get_figure()
    fig.savefig('Word_Count_dist.png')
    plt.clf()

    # For the top and bottom n rated restaurants what in general were they serving?
    avg_ratings = data.groupby("restaurant_name").agg(review_avg=("rating_review", "mean"),
                                                      reviews=("rating_review", "count")).\
        sort_values(by=['review_avg',  'reviews'], ascending=False)

    # High Trust Per review
    top_ten_ht = avg_ratings.loc[avg_ratings['reviews'] > 100, :].head(10)
    bottom_ten_ht = avg_ratings.loc[avg_ratings['reviews'] > 100, :].tail(10)

    # Low Trust Per review
    top_ten_lt = avg_ratings.loc[avg_ratings['reviews'] > 500, :].head(5)
    top_five = top_ten_lt.index.to_list()
    top_five = [('top', x) for x in top_five]

    bottom_ten_lt = avg_ratings.loc[avg_ratings['reviews'] > 500, :].tail(5)
    bottom_five = bottom_ten_lt.index.to_list()
    bottom_five = [('bottom', x) for x in bottom_five]
    rests_breakdown = {}
    food = wn.synset('food.n.02')
    dishes = wn.synset('dish.n.02')
    service = wn.synset('service.n.02')
    food = list(set([w for s in food.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    food += list(set([w for s in dishes.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
    for food_place in bottom_five+top_five:
        test_data = data.loc[data['restaurant_name'] == food_place[1], :]
        flat_list = [item for sublist in test_data['lem_reviews'] for item in sublist]
        food_list = [item for item in flat_list if item in food]
        nlp_words = nltk.FreqDist(food_list)
        wcloud = WordCloud().generate_from_frequencies(nlp_words)
        plt.imshow(wcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"WordClouds\\{food_place[0]}_{food_place[1]}_Wordcloud.png")
        plt.clf()

    # For Reviewers with <3, 3-10 and 10+ same questions
    # what was their average review legnth.
    audience_grouping = data.groupby('author_id').agg(reviews=("rating_review", "count"),
                             avg_word_count=("word_count", "mean"),
                             avg_review_score=("rating_review", "mean"))

    review_count = audience_grouping.groupby('reviews').agg(count=('reviews', 'count'))
    inter = review_count.sort_index().cumsum()
    inter['perc'] = inter['count'] / audience_grouping.shape[0]  #
    aex = inter['perc'].plot(figsize=(20, 15))
    fig = aex.get_figure()
    plt.axvline(x=1)
    plt.axvline(x=5)
    plt.axvline(x=10)
    fig.savefig('test.png')
    fig.clf()


    audience_review_len_groupings = np.quantile(audience_grouping['avg_word_count'], [0, .33, .66, 1])
    audience_grouping['word_count_group'] = pd.cut(audience_grouping['avg_word_count'],
                                                   audience_review_len_groupings, labels=['low', 'med', 'high'])
    audience_grouping['review_count_group'] = pd.cut(audience_grouping['reviews'],
                                                   [0, 3, 10, max(audience_grouping['avg_word_count'])],
                                                     labels=['0-3', '3-10', '10+'])
    audience_grouping.groupby('word_count_group').agg(reviews=("reviews", "count"),
                                                      avg_review_score=("avg_review_score", "mean"),
                             avg_word_count=("avg_word_count", "mean"))

    rc_scores = audience_grouping.groupby('review_count_group').agg(reviews=("reviews", "count"),
                                                      avg_review_score=("avg_review_score", "mean"),
                                                      avg_word_count=("avg_word_count", "mean"))['avg_review_score']

    sns.set(rc={'figure.figsize': (10, 6)})
    bplot = sns.barplot(x='review_count_group', y='avg_review_score', data=rc_scores.reset_index())
    bplot.tick_params(labelsize=22)
    bplot.set_xlabel("Review Count", fontsize=30)
    bplot.set_ylabel("Average Review Score", fontsize=30)
    sns.set_context("poster")
    fig = bplot.get_figure()
    fig.savefig('test3.png')
    fig.clf()
