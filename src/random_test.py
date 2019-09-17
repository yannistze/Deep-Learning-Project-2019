import pandas as pd
import numpy as np
from sklearn import metrics

def evaluate_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')

    print(
        "Accuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}\n\n" \
            .format(accuracy, precision, recall, f1)
    )

df = pd.read_csv('../data/final/artist_25/test.tsv', delimiter='\t',
                 dtype=str)
df.columns = ['artist', 'lyrics']

genre_truth = df.artist.apply(lambda x: x.index('1')).values
genre_rando = np.random.randint(0, 10, len(genre_truth))
genre_majority = np.array([2] * len(genre_truth))

print('Random Prediction:\n')
evaluate_metrics(genre_truth, genre_rando)

print('Majority Prediction:\n')
evaluate_metrics(genre_truth, genre_majority)

# classes = ['Country', 'Electronic', 'Folk', 'Hip-Hop', 'Indie',
#           'Jazz', 'Metal', 'Pop', 'R&B', 'Rock']
# print(metrics.classification_report(genre_target, genre_preds, digits=3))