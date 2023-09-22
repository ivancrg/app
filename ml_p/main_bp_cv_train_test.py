import numpy as np
from mlp_keras import MLP
import tensorflow as tf
import pandas as pd

PREDICTING_INTEREST = '/postop_binary'

OPTIMIZER = 'SGD'
LEARNING_RATE = 0.1
DROPOUT = 0.05
LOSS = 'categorical_crossentropy'
BINARY = ('binary' in PREDICTING_INTEREST)
OVERSAMPLING = True

BASE_FOLDER = './report/OS_NN' if OVERSAMPLING else './report/NO_OS'
FOLDER = BASE_FOLDER + PREDICTING_INTEREST
FILE = '/data_norm.csv'

mlp = MLP(FOLDER + FILE)
log = pd.DataFrame()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True)
rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

mlp.cv(
    9,
    2 if BINARY else 3,
    optimizer_text=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    callbacks=[early_stopping_callback, rlr_callback],
    dropout=DROPOUT,
    visualization_save_folder=FOLDER,
    oversampling=OVERSAMPLING,
    loss=LOSS
)

mlp.train(
    9,
    2 if BINARY else 3,
    optimizer_text=OPTIMIZER,
    learning_rate=0.1,
    callbacks=[early_stopping_callback, rlr_callback],
    dropout=0.24,
    save_folder=FOLDER,
    oversampling=OVERSAMPLING,
    test=True
)