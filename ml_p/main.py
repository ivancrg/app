import numpy as np
from ml_p.mlp import MLP
import tensorflow as tf
import pandas as pd

"""
    Class that is used to run GS using losses,
    optimizers, learning_rates and dropouts as input.
"""
class MLPCV():
    def __init__(self, folder, file, oversampling=False, binary=True):
        self.folder = folder
        self.file = file
        self.oversampling = oversampling
        self.binary = binary

    def run(self):
        losses = ['categorical_crossentropy', 'binary_crossentropy'] if self.binary else [
            'sparse_categorical_crossentropy', 'categorical_crossentropy']
        optimizers = ['AdamW',
                      'Adam',
                      'SGD',
                      # 'Adafactor',
                      # 'Nadam'
                      ]
        learning_rates = [10 ** (-i) for i in range(1, 6)]
        dropouts = np.linspace(0.05, 0.2, 4)

        print(self.folder)
        print(losses)
        print(optimizers)
        print(learning_rates)
        print(dropouts)

        mlp = MLP(self.folder + self.file)
        log = pd.DataFrame()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True)
        rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

        best = (0, 1, '')
        i = 0
        for loss in losses:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for dropout in dropouts:
                        # Getting results from one iteration of GS using CV
                        average_val_loss, average_val_accuracy, train_accuracies, train_losses, val_losses, val_accuracies, names = mlp.cv(
                            optimizer, learning_rate, [early_stopping_callback, rlr_callback], dropout=dropout, visualization_save_folder=None, oversampling=self.oversampling, loss=loss)

                        if average_val_accuracy > best[0]:
                            best = average_val_accuracy, average_val_loss, names

                        # Logging the results
                        log = pd.concat([log,
                                        pd.DataFrame({
                                            'train_accuracy': train_accuracies,
                                            'train_loss': train_losses,
                                            'val_accuracy': val_accuracies,
                                            'val_loss': val_losses,
                                            'average_val_accuracy': [average_val_accuracy for _ in range(len(val_accuracies))],
                                            'average_val_loss': [average_val_loss for _ in range(len(val_losses))],
                                            'name': names
                                        })],
                                        axis=0
                                        )

                        i += 1
                        print(val_accuracies, names)
                        print(best)
                        print(
                            f'Progress: {i}/{len(losses) * len(optimizers) * len(learning_rates) * len(dropouts)} ({round(i / (len(losses) * len(optimizers) * len(learning_rates) * len(dropouts)) * 100, 2)}%)')

        log.to_csv(f'{self.folder}/tmp_nn_gs_log.csv', index=False)

        log.sort_values(by=['average_val_accuracy',
                        'average_val_loss'], ascending=False)
        best_params = bp = log.iloc[0]['name'].split('_')
        bp_loss = bp[-5] + '_' + bp[-4] + '_' + \
            bp[-3] if bp[-5] == 'sparse' else bp[-4] + '_' + bp[-3]
        bp_opt, bp_lr, bp_inl, bp_hidl, bp_dp = str(bp[3]), float(bp[-2]), int(bp[0]), [int(i) for i in bp[1][1:-1].split(', ')], float(bp[2])

        # Training and testing using the best
        # results gathered through GS
        mlp.train(
            optimizer_text=bp_opt,
            learning_rate=bp_lr,
            callbacks=[early_stopping_callback, rlr_callback],
            input_layer=bp_inl,
            hidden_layers=bp_hidl,
            dropout=bp_dp,
            save_folder=self.folder,
            test=True,
            loss=bp_loss
        )
