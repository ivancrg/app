# Data visualization
import matplotlib.pyplot as plt
import numpy as np
import display_data as dd

import tensorflow as tf

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Train-Test
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 14})


class MLP():
    def __init__(self, save_folder=None):
        self.save_folder = save_folder
        self.optimizer_text = 'Adam'
        self.learning_rate = 0.001
        self.callbacks = []
        self.input_layer = 512
        self.hidden_layers = [512, 256, 128]
        self.dropout = 0.2
        self.loss = 'categorical_crossentropy'

        self.classifier = None
        self.history = None

        # # PDP sklearn things
        # self._estimator_type = 'classifier'
        # self.classes_ = data.iloc[:, -1].unique()

    # # PDP sklearn thing
    # def fit(self, args):
    #     self.train(args)

    # # PDP sklearn thing
    # def __sklearn_is_fitted__(self):
    #     return True

    # # PDP sklearn thing
    # def predict_proba(self, X):
    #     return self.predict(X, np_classes=False)

    # def __call__(self, X):
    #     return self.predict(X)

    def set_params(self, params):
        for key, val in params.items():
            if not hasattr(self, key):
                print(f'vc::ml_p::mlp.py::set_params {key}:No such attribute!')
                continue

            setattr(self, key, val)

    def get_params(self):
        return {
            'optimizer': self.optimizer_text,
            'learning_rate': self.learning_rate,
            'callbacks': self.callbacks,
            'input_layer': self.input_layer,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'loss': self.loss,
        }

    def get_optimizer(self, optimizer_text, learning_rate):
        if optimizer_text == 'RMSprop':
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_text == 'SGD':
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_text == 'AdamW':
            optimizer = tf.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer_text == 'Adadelta':
            optimizer = tf.optimizers.Adadelta(learning_rate=learning_rate)
        elif optimizer_text == 'Adagrad':
            optimizer = tf.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_text == 'Adamax':
            optimizer = tf.optimizers.Adamax(learning_rate=learning_rate)
        elif optimizer_text == 'Adafactor':
            optimizer = tf.optimizers.Adafactor(learning_rate=learning_rate)
        elif optimizer_text == 'Nadam':
            optimizer = tf.optimizers.Nadam(learning_rate=learning_rate)
        elif optimizer_text == 'Ftrl':
            optimizer = tf.optimizers.Ftrl(learning_rate=learning_rate)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        return optimizer

    def get_name(self):
        return f'{self.input_layer}_{self.hidden_layers}_{self.dropout}_{self.optimizer_text}_{self.loss}_{self.learning_rate}'

    def create_model(self, n_input_features, output_classes, input_layer, hidden_layers, dropout, non_cat=False):
        model = Sequential()
        model.add(Dense(input_layer, input_shape=(
            n_input_features,), activation="relu"))

        for neurons in hidden_layers:
            model.add(Dense(neurons, activation="relu"))
            model.add(Dropout(dropout))

        if non_cat:
            model.add(Dense(1, activation="sigmoid"))
        else:
            model.add(Dense(output_classes, activation="softmax"))

        return model

    def plot_histories(self, history, location):
        # Plot training and validation losses
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracies
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(location)

    def fit(self, X, y):
        n_input_features = X.shape[1]
        n_outputs = len(np.unique(y))

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True)

        model = self.create_model(
            n_input_features, n_outputs, self.input_layer, self.hidden_layers, self.dropout, non_cat=(self.loss != 'categorical_crossentropy'))

        model.compile(
            loss=self.loss,
            optimizer=self.get_optimizer(
                self.optimizer_text, self.learning_rate),
            metrics=["accuracy"]
        )

        # One-hot encoded multiclass problem with sparse_cat_crossentropy
        if self.loss != 'categorical_crossentropy' and len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
            y_valid = np.argmax(y_valid, axis=1)

        # Binary to categorical output
        if n_outputs <= 2 and self.loss == 'categorical_crossentropy':
            y_train = tf.keras.utils.to_categorical(y_train, n_outputs)
            y_valid = tf.keras.utils.to_categorical(y_valid, n_outputs)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=300,
            batch_size=16,
            callbacks=self.callbacks,
            verbose=1
        )

        self.classifier = model
        self.history = history
    
    def fit_cv(self, X_train, y_train, X_valid, y_valid, fig_location=None, verbose=False):
        n_input_features = X_train.shape[1]
        n_outputs = len(np.unique(y_train))

        print(f'inputfeat {n_input_features}, nout {n_outputs}')

        model = self.create_model(
            n_input_features, n_outputs, self.input_layer, self.hidden_layers, self.dropout, non_cat=(self.loss != 'categorical_crossentropy'))

        model.compile(
            loss=self.loss,
            optimizer=self.get_optimizer(
                self.optimizer_text, self.learning_rate),
            metrics=["accuracy"]
        )

        # One-hot encoded multiclass problem with sparse_cat_crossentropy
        if self.loss != 'categorical_crossentropy' and len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
            y_valid = np.argmax(y_valid, axis=1)
            print('argmax')

        if verbose:
            model.summary()

        # Binary to categorical output
        if n_outputs <= 2 and self.loss == 'categorical_crossentropy':
            y_train = tf.keras.utils.to_categorical(y_train, n_outputs)
            y_valid = tf.keras.utils.to_categorical(y_valid, n_outputs)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=300,
            batch_size=16,
            callbacks=self.callbacks,
            verbose=verbose
        )

        train_acc = history.history['accuracy'][np.argmax(
            history.history['val_accuracy'])]
        train_loss = history.history['loss'][np.argmax(
            history.history['val_accuracy'])]
        val_acc = np.max(history.history['val_accuracy'])
        val_loss = history.history['val_loss'][np.argmax(
            history.history['val_accuracy'])]
        
        tf.keras.backend.clear_session()

        if fig_location is not None:
            self.plot_histories(history, fig_location)
        
        return (train_acc, train_loss, val_acc, val_loss)

    def test(self, X, y):
        if self.save_folder is not None:
            self.classifier.save(
                f'{self.save_folder}/nn_{self.get_name()}_save')

            y_pred = self.classifier.predict(X)
            dd.visualize_cr_cm(np.argmax(y, axis=1), np.argmax(
                y_pred, axis=1), self.save_folder, f'nn_{self.get_name()}')

    def predict(self, X, classes=True):
        if self.classifier is None:
            print("mlp.py::predict::No trained classifier!")
            return

        y_pred = self.classifier.predict(X)

        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X, classes=True):
        if self.classifier is None:
            print("mlp.py::predict::No trained classifier!")
            return

        y_pred = self.classifier.predict(X)
        return y_pred
