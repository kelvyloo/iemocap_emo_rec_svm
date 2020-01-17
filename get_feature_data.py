#!/usr/bin/python3

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

csv_filepath = "/home/kelvin/c487/data/features/csvs/"
label_file   = "/home/kelvin/c487/data/ground_truth.csv"

csvs = os.listdir(csv_filepath)
csvs.sort()

total       = len(csvs)
percent     = int(round(total * 0.01))
i           = 0
progress    = i/total * 100

prog_template = "Gathering feature data...{:.3f}%"
time_template = "(Time elapsed: {:.3f}s)"

filenames = []
emotions  = []

classes        = np.empty((total, ))
training_data  = np.empty((total, 6375))
training_class = np.empty((total, ))
predict_data   = np.empty((total, 6375))
predict_class  = np.empty((total, ))

emotion_counter = {1: 0, # Neutral State
                   2: 0, # Sadness
                   3: 0, # Anger
                   4: 0} # Happiness

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Load in ground truth data
with open(label_file, 'r') as labels:
    for row in labels.readlines():
        col = row.split(',')

        filenames.append(col[0])
        emotions.append(col[1])
        emo_enum = col[2][0]

        try:
            classes[i] = int(emo_enum)
            i += 1
        except ValueError:
            continue

# Extract feature data from csv files
i = 0
t0 = time.time()

print(prog_template.format(progress))

for csv in csvs:
    data = np.genfromtxt(csv_filepath+csv, delimiter=',')

    if emotion_counter[classes[i]] % 2:
        training_data[i] = data[1]
        training_class[i] = classes[i]
    else:
        predict_data[i] = data[1]
        predict_class[i] =  classes[i]

    emotion_counter[classes[i]] += 1
    i += 1

    if (i % percent == 0):
        progress = i/total * 100
        print(prog_template.format(progress))

elapsed_t = time.time() - t0
print("Done " + time_template.format(elapsed_t))

# Train SVM model
print("Training SVM...")
t0 = time.time()

training_data = training_data[:, ~np.isnan(training_data).any(axis=0)]
classifier = svm.SVC(gamma = 'scale')
classifier.fit(training_data, classes)

elapsed_t = time.time() - t0
print("Training Finished " + time_template.format(elapsed_t))

# Predict
print("Classifying data with SVM model...")
t0 = time.time()

predict_data = predict_data[:, ~np.isnan(predict_data).any(axis=0)]
predicted = classifier.predict(predict_data)

elapsed_t = time.time() - t0
print("Prediction Complete " + time_template.format(elapsed_t))

# Plot confusion matrix
plot_confusion_matrix(predict_class, predicted, classes)
plt.show()
