import datetime
import pickle
import random

import numpy as np
import pandas as pd
import implicit
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from scipy import sparse

import metrics
import plots
import models
import util
import logging

# Seed
random_state = 42
random.seed(random_state)

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d %H-%M-%S")

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(f"log/IMPLICIT_{now}.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# GLOBAL VARIABLE
NUM_RECOMMEND = 20

# Load Data
df = pd.read_csv("./data/order_20200921.csv", encoding="utf-8-sig")
logger.info(df.head())

# Preprocess
df = df.drop_duplicates(subset=["CUST_NO", "ITEM_CD"])
df["ORD_CNT"] = 1

# Most Popular Items
most_popular = util.most_popular_periods(df, 14)

with open('res/most_popular.txt', 'wb') as f:
    pickle.dump(most_popular, f)

with open('res/most_popular.txt', 'rb') as f:
    most_popular = pickle.load(f)

logger.info(f"\nMost Popular : {most_popular[:5]}")

# DF to interaction
df = pd.pivot_table(df, values='ORD_CNT', index="CUST_NO", columns="ITEM_CD", aggfunc=np.sum, fill_value=0.0)
logger.info(f"Maximum Value : {df.to_numpy().max()}")
logger.info(f"Shape : {df.shape}")

# Filter items which bought less 2 times
user_cs_item = df.sum(axis=0) > 2
# Filter items which bought less 2 times
user_cs = df.sum(axis=1) > 2

df = df[df.columns[user_cs_item]]
df = df[user_cs]

logger.info(f"Columns : {list(df.columns[:5])}")
logger.info(f"Indices : {list(df.index[:5])}")
logger.info(f"Shape : {df.shape}")

# Shuffle indices
df = shuffle(df, random_state=random_state)
# Shuffle columns
df = shuffle(df.T, random_state=random_state).T

# Item Catalog
items = list(range(len(df.columns)))

# DF to numpy array
df = df.to_numpy().astype(np.float32)
logger.info(f"Shape : {df.shape}")

# To CSR matrix
sparse_df = sparse.csr_matrix(df)

# # Split datas
# train, test = train_test_split(df, test_size=0.2, random_state=random_state)

ans = []
for row in df:
    tmp = [i for i, val in enumerate(row) if val == 1]
    ans.append(tmp)
print("ans")
print(ans[:5])

ans_sparse = []
for row in ans:
    tmp = [0] * len(items)
    for i in tmp:
        tmp[i] = 1
    ans_sparse.append(tmp)
ans_sparse = np.array(ans_sparse)

# item/user/confidence
# (7124, 9217)
print(sparse_df.T.shape)

model = implicit.als.AlternatingLeastSquares(factors=50, use_gpu=False)
model.fit(sparse_df.T)

_, test = train_test_split(list(range(len(df))), test_size=0.2, random_state=random_state)

pred = []
for i in test:
    tmp = model.recommend(userid=i, user_items=sparse_df, N=NUM_RECOMMEND)
    pred.append(tmp)
pred = np.array(pred)

pred = pred[:, :, 0]
pred = pred.astype(int)
print("pred")
print(pred.shape)
print(pred[:5])

print("test")
test = df[test, :]
print(test.shape)
print(test[:5])

pred_sparse = []
for row in pred:
    tmp = [0] * len(items)
    for i in tmp:
        tmp[i] = 1
    pred_sparse.append(tmp)
pred_sparse = np.array(pred_sparse)

logger.info(f"MAP_K : {metrics.mapk(ans, pred, NUM_RECOMMEND)}")
logger.info(f"MAR_K : {metrics.mark(ans, pred, NUM_RECOMMEND)}")
logger.info(f"NOVELTY : Do not measure")
logger.info(f"MSE : {metrics.mse(test, pred_sparse)}")
logger.info(f"RMSE : {metrics.rmse(test, pred_sparse)}")
logger.info(f"PRECISION : {metrics.recommender_precision(pred, ans)}")
logger.info(f"RECALL : {metrics.recommender_recall(pred, ans)}")
logger.info(f"PERSONALIZATION : {metrics.personalization(pred)}")
logger.info(f"PREDICTION COVERAGE : {metrics.prediction_coverage(pred, items)}")

# plots.long_tail_plot(df, "ITEM_CD", "purchases", 0.2, False)