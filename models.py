import datetime
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Concatenate, Lambda
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam


def get_ae(
        num_items: int,
        hidden_dim: int,
        lr: float):
    item_input = Input(num_items,)
    dropped_input = Dropout(0.5)(item_input)
    hidden = Dense(hidden_dim, kernel_initializer='glorot_uniform', activation='sigmoid')(dropped_input)
    item_output = Dense(num_items, kernel_initializer='glorot_uniform', activation='sigmoid')(hidden)
    model = Model(item_input, item_output)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')
    return model