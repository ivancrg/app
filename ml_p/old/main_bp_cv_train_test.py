import numpy as np
from ml_p.mlp import MLP
import tensorflow as tf
import pandas as pd

PREDICTING_INTEREST = '/SEV'

OPTIMIZER='SGD'
LEARNING_RATE=0.1
INPUT_LAYER=512
HIDDEN_LAYERS=[512, 256, 128]
DROPOUT=0.2
LOSS='categorical_crossentropy'
BINARY = True
# BINARY = ('binary' in PREDICTING_INTEREST)
OVERSAMPLING = True

BASE_FOLDER = './report'
# BASE_FOLDER = './report/OS_NN' if OVERSAMPLING else './report/NO_OS'
FOLDER = BASE_FOLDER + PREDICTING_INTEREST
FILE = '/data_norm.csv'

mlp = MLP(FOLDER + FILE)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True)
rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

mlp.train(
    optimizer_text=OPTIMIZER,
    learning_rate=0.1,
    callbacks=[early_stopping_callback, rlr_callback],
    dropout=0.24,
    save_folder=FOLDER,
    oversampling=OVERSAMPLING,
    test=True
)