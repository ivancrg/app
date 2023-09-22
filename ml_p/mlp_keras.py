# Data visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import display_data as dd

import tensorflow as tf

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Train-Test
from sklearn.model_selection import train_test_split, StratifiedKFold

plt.rcParams.update({'font.size': 14})

class MLP():
    def __init__(self, filename):
        data = pd.read_csv(filename)
        self.X, self.y = data.iloc[:, :-1], pd.DataFrame(data.iloc[:, -1])

        y_cat = to_categorical(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values, y_cat, stratify=y_cat, test_size=0.2, random_state=42, shuffle=True)

        self.trained_model = None
        
        # PDP sklearn things
        self._estimator_type = 'classifier'
        self.classes_ = data.iloc[:, -1].unique()

    # PDP sklearn thing
    def fit(self, args):
        self.train(args)

    # PDP sklearn thing
    def __sklearn_is_fitted__(self):
        return True
    
    # PDP sklearn thing
    def predict_proba(self, X):
        return self.predict(X, np_classes=False)

    def __call__(self, X):
        return self.predict(X)
    
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

    def create_model(self, input_features, output_classes, input_layer, hidden_layers, dropout, binary=False):
        model = Sequential()
        model.add(Dense(input_layer, input_shape=(
            input_features,), activation="relu"))

        for neurons in hidden_layers:
            model.add(Dense(neurons, activation="relu"))
            model.add(Dropout(dropout))

        model.add(Dense(output_classes, activation="softmax"))

        if binary:
            model.add(Dense(1, activation="softmax"))

        return model

    def oversample(self, df):
        output_name = df.columns[-1]

        classes_counts = [(c, sum(df.iloc[:, -1] == c))
                    for c in df.iloc[:, -1].unique()]
        classes_counts.sort(key=lambda x: x[1])

        max_cnt = classes_counts[-1][-1]

        for cls, cnt in classes_counts:
            if cnt == max_cnt:
                continue
            
            tmp = df[df[output_name] == cls].copy()

            tdf = pd.DataFrame()
            for _ in range(max_cnt // cnt - 1):
                tdf = pd.concat([tdf, tmp.copy()], ignore_index=True)
            tdf = pd.concat([tdf, tmp.iloc[:(max_cnt - cnt - len(tdf))]], ignore_index=True)

            df = pd.concat([df, tdf], ignore_index=True)
        
        # unique_rows_with_zero = df[df[output_name] == 0].drop_duplicates().shape[0]
        # print(f"Number of unique rows with 0 in the last column: {unique_rows_with_zero}")
        # unique_rows_with_one = df[df[output_name] == 1].drop_duplicates().shape[0]
        # print(f"Number of unique rows with 1 in the last column: {unique_rows_with_one}")
        # print(df[output_name].value_counts())
        
        df = df.sample(frac=1, random_state=42)

        return df

    def cv(self, input_features, outputs, optimizer_text='Adam', learning_rate=0.001, callbacks=[], input_layer=512, hidden_layers=[512, 256, 128], dropout=0.2, visualization_save_folder=None, oversampling=False, loss='categorical_crossentropy'):
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        train_accuracies = []
        train_losses = []
        val_accuracies = []
        val_losses = []
        names = []

        for fold_idx, fold in enumerate(skf.split(self.X_train, np.argmax(self.y_train, axis=1))):
            train_index, val_index = fold
            X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
            y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]

            if oversampling:
                X_train_fold_df = pd.DataFrame(X_train_fold, columns=self.X.columns)
                y_train_fold_df = pd.DataFrame(np.argmax(y_train_fold, axis=1), columns=self.y.columns)
                train_fold = pd.concat([X_train_fold_df, y_train_fold_df], axis=1)
                train_fold = self.oversample(train_fold)
                X_train_fold, y_train_fold = train_fold.iloc[:, :-1].to_numpy(), pd.get_dummies(train_fold.iloc[:, -1]).astype(float).to_numpy()

            model = self.create_model(
                input_features, outputs, input_layer, hidden_layers, dropout, binary=(loss=='binary_crossentropy'))
            model.summary()

            model.compile(
                loss=loss,
                optimizer=self.get_optimizer(optimizer_text, learning_rate),
                metrics=["accuracy"]
            )
            
            if loss != 'categorical_crossentropy':
                y_train_fold = np.argmax(y_train_fold, axis=1)
                y_val_fold = np.argmax(y_val_fold, axis=1)

            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=300,
                batch_size=16,
                callbacks=callbacks,
                verbose=0
            )

            train_accuracy = history.history['accuracy'][np.argmax(
                history.history['val_accuracy'])]
            train_loss = history.history['loss'][np.argmax(
                history.history['val_accuracy'])]
            val_accuracy = np.max(history.history['val_accuracy'])
            val_loss = history.history['val_loss'][np.argmax(
                history.history['val_accuracy'])]

            name = f'{input_layer}_{hidden_layers}_{dropout}_{optimizer_text}_{loss}_{learning_rate}_{fold_idx}'
            # model.save(f'{visualization_save_folder}/{name}')

            # Append the accuracy and loss to the lists
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)
            names.append(name)

        if visualization_save_folder is not None:
            kf_scores = {'test_score': np.array(val_accuracies)}
            dd.visualize_cv(n_splits, kf_scores,
                            visualization_save_folder, f'nn_{name}_')

        # Calculate the average accuracy and loss on the validation set for cross-validation
        average_val_accuracy = np.mean(val_accuracies)
        average_val_loss = np.mean(val_losses)

        return (average_val_loss, average_val_accuracy, train_accuracies, train_losses, val_losses, val_accuracies, names)

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

    def train(self, input_features, outputs, optimizer_text='Adam', learning_rate=0.001, callbacks=[], input_layer=512, hidden_layers=[512, 256, 128], dropout=0.2, save_folder=None, test=False, oversampling=False, loss='categorical_crossentropy'):
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.X_train, self.y_train, stratify=self.y_train, test_size=0.2, random_state=42, shuffle=True)
        
        if oversampling:
            X_train_df = pd.DataFrame(X_train, columns=self.X.columns)
            y_train_df = pd.DataFrame(np.argmax(y_train, axis=1), columns=self.y.columns)
            train_df = pd.concat([X_train_df, y_train_df], axis=1)
            train_df = self.oversample(train_df)
            X_train, y_train = train_df.iloc[:, :-1].to_numpy(), pd.get_dummies(train_df.iloc[:, -1]).astype(float).to_numpy()

        model = self.create_model(
                input_features, outputs, input_layer, hidden_layers, dropout, binary=(loss=='binary_crossentropy'))

        model.compile(
            loss=loss,
            optimizer=self.get_optimizer(optimizer_text, learning_rate),
            metrics=["accuracy"]
        )

        if loss != 'categorical_crossentropy':
            y_train = np.argmax(y_train, axis=1)
            y_valid = np.argmax(y_valid, axis=1)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=300,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        self.trained_model = model

        if save_folder is not None:
            name = f'{input_layer}_{hidden_layers}_{dropout}_{optimizer_text}_{loss}_{learning_rate}'
            model.save(f'{save_folder}/nn_save_{name}')
            self.plot_histories(history, f'{save_folder}/nn_histories{name}.png')

            if test is True:
                y_pred = model.predict(self.X_test)
                dd.visualize_cr_cm(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1), save_folder, f'nn_{name}_')

    def predict(self, X, np_classes=True):
        if self.trained_model is None:
            print("mlp_keras.py::predict::No trained model!")

        y_pred = self.trained_model.predict(X)

        if np_classes:
            return np.argmax(y_pred, axis=1)

        return y_pred