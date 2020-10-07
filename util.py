import datetime
from collections import defaultdict

import pandas as pd


def most_popular(train_df):
    '''
    train_df
        rows    : users
        columns : items
        vlaues  : buy(1) or not(0)
    return : most popular 20 items
    '''
    my_sum_serie = train_df.sum().sort_values(ascending=False)[:20]
    my_list = my_sum_serie.tolist()
    my_dict = my_sum_serie.to_dict()
    return list(my_dict.keys())


def most_popular_periods(df, t):
    '''
    df
        ITEM_CD     : 상품 코드
        ORD_REQ_DTM : 주문 시각
    return : most popular 100 items
    '''
    df['ORD_REQ_DTM'] = pd.to_datetime(df['ORD_REQ_DTM'])
    crit = df.iloc[-1]["ORD_REQ_DTM"] - datetime.timedelta(days=t)
    df = df[df["ORD_REQ_DTM"] > crit]
    popular = list(df["ITEM_CD"].value_counts()[:100].keys())
    popular = [str(i) for i in popular]
    return popular


def remove_seen(seen, l):
    seen = set(seen)
    return [x for x in l if not (x in seen)]


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n