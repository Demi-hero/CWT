import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = pd.read_csv("Datasets\\Cleaned_reviews.csv")

# As this is just a test case I have removed users who have less than 6 reviews.
# To save memory and allow it to be run locally.
# Still gives us around 315K rows to play with

audience_grouping = data.groupby('author_id').agg(reviews=("rating_review", "count"))
audience_grouping = audience_grouping.loc[audience_grouping["reviews"] > 5, :]

small_data = data.merge(audience_grouping, on='author_id', how='inner')
reader = Reader(rating_scale=(1, 5))

dataset = Dataset.load_from_df(small_data[["author_id", "restaurant_name", "rating_review"]], reader)

sim_options = {
    "name": ["msd"],
    "min_support": [3, 5],
    "user_based": [True],
}

param_grid = {"sim_options": sim_options}

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3, refit=True)
gs.fit(dataset)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])
pred = gs.predict("UID_100062", "The_Piano_Works_Farringdon")
print(pred.est)

sim_options = gs.best_params["rmse"]['sim_options']
algo = KNNWithMeans(sim_options=sim_options)
algo.fit(dataset)
# Improvements could be to create an associated dataset of the common food items or resteraunt types
# Then we could use some of the more advanced options like matrix decomposition algorithm
