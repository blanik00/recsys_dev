import datetime
import pickle
import random
import collections

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

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
file_handler = logging.FileHandler(f"log/AE_{now}.log")
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

# Split datas
train, test = train_test_split(df, test_size=0.2, random_state=random_state)

item_list = []
for row in train:
    tmp = [i for i, val in enumerate(row) if val == 1]
    item_list.extend(tmp)

item_counter = collections.Counter(item_list)

ans = []
for row in test:
    tmp = [i for i, val in enumerate(row) if val == 1]
    ans.append(tmp)

logger.info(f"Train Shape : {train.shape}")
logger.info(f"Test Shape : {test.shape}")

model = models.get_ae(df.shape[1], 256, 0.001)
model.summary()

# # Train
# cb_checkpoint = ModelCheckpoint(filepath='./res/user_model_20200921.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
# model.fit(x=train, y=train, batch_size=32, epochs=100, verbose=2, validation_split=0.2, shuffle=True, callbacks=[cb_checkpoint])

model.load_weights("res/user_model_20200921.hdf5")
pred = model.predict(test).argsort(axis=1)[:, ::-1][:, :NUM_RECOMMEND]

pred_sparse = []
for row in pred:
    tmp = [0] * len(items)
    for i in tmp:
        tmp[i] = 1
    pred_sparse.append(tmp)
pred_sparse = np.array(pred_sparse)

logger.info(f"MAP_K : {metrics.mapk(ans, pred, NUM_RECOMMEND)}")
logger.info(f"MAR_K : {metrics.mark(ans, pred, NUM_RECOMMEND)}")
logger.info(f"NOVELTY : {metrics.novelty(pred, item_counter, train.shape[0], NUM_RECOMMEND)[0]}")
logger.info(f"MSE : {metrics.mse(test, pred_sparse)}")
logger.info(f"RMSE : {metrics.rmse(test, pred_sparse)}")
logger.info(f"PRECISION : {metrics.recommender_precision(pred, ans)}")
logger.info(f"RECALL : {metrics.recommender_recall(pred, ans)}")
logger.info(f"PERSONALIZATION : {metrics.personalization(pred)}")
logger.info(f"PREDICTION COVERAGE : {metrics.prediction_coverage(pred, items)}")

# plots.long_tail_plot(df, "ITEM_CD", "purchases", 0.2, False)