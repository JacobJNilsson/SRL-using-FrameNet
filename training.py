from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime


def train_svm(data: List[dict], file: str = None, gamma: float = 0.001, c: int = 100):
    # Just to make sure things are good and random
    random.shuffle(data)

    # Vectorize data
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()

    # Flatten the images
    n_samples = len(data)
    half = int(n_samples / 2)
    # data = data.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma, C=c)

    # Split the test data
    X = [p[:-1] for p in data]
    y = [p[-1] for p in data]
    X_train = X[:half]
    X_test = X[half:]
    y_train = y[:half]
    y_test = y[half:]

    # Learn the data on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    if file:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f = open("arg_ident_results.txt", "a")
        f.write(f"{dt_string}:\n{disp.confusion_matrix}\n\n")
        f.close()