import pandas as pd
from IPython import display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def visualize_cv(n_splits, scores, save_folder=None, prefix=''):
    for fold, score in enumerate(scores['test_score']):
        print(f"Fold {fold+1} accuracy: {score}")
    accuracy_mean = scores['test_score'].mean()

    print(f"Average accuracy: {accuracy_mean}")

    plt.figure(1, figsize=(10, 6))
    bars = plt.bar(range(1, n_splits + 1), scores['test_score'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval,
                round(yval, 5), ha='center', va='bottom')

    plt.axhline(y=accuracy_mean, color='r', linestyle='--',
                label='Mean accuracy value')

    plt.xlabel('Fold')
    plt.ylabel('Accuracy')

    plt.ylim(0, 1)
    plt.legend()
    if save_folder is not None:
        plt.savefig(f'{save_folder}/{prefix}cv.png')
    plt.show()

def visualize_cr_cm(y_true, y_pred, save_folder=None, prefix=''):
    cr = classification_report(y_true, y_pred)
    if save_folder is not None:
        with open(f'{save_folder}/{prefix}classification_report.txt', 'w') as f:
            print(cr, file=f) 

    crdf = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
    crdf.iloc[:-1, :-3].T.plot(kind='bar')
    if save_folder is not None: plt.savefig(f'{save_folder}/{prefix}classification_report.png')
    plt.show()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    if save_folder is not None: plt.savefig(f'{save_folder}/{prefix}confusion_matrix.png')
    plt.show()