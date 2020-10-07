from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

import util

# GLOBAL VARIABLE
NUM_RECOMMEND = 20

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.2)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

top_n = util.get_top_n(predictions, n=NUM_RECOMMEND)

for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])