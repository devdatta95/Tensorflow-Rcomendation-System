import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from config import *



# Ratings data.
ratings = tfds.load(RATINGS_DATA, split="train")
# Features of all the available movies.
movies = tfds.load(MOVIES_DATA, split="train")

print("###################### RATING DATA #######################")
for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)
print("####################### MOVIE DATA ######################")
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

# we will use rating data for retrieval systm
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

# train test split
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print("[INFO] Top 10 Unique Movie Titles: ", unique_movie_titles[:10])