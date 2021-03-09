from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split

from datetime import datetime


def train_svm(x_data: List, y_data: List, gamma: float = 0.001, c: float = 100):

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma, C=c)

    # Learn the data on the train subset
    clf.fit(x_data, y_data)

    return clf


def test_svm(
    clf, x_test: List, y_test: List, file: str = None, display_conf_matrix: bool = False
):

    # Predict the value of the digit on the test subset
    predicted = clf.predict(x_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    if file:
        f = open(file, "a")

        if display_conf_matrix:
            # This takes a crazy long time, why?
            disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
            disp.figure_.suptitle("Confusion Matrix")
            print(f"Confusion matrix:\n{disp.confusion_matrix}")
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            f.write(f"{dt_string}:\n{disp.confusion_matrix}\n\n")
            f.close()
        else:
            f.write(f"Classification report for classifier {clf}:\n")
            f.write(f"{metrics.classification_report(y_test, predicted)}\n")
            f.close()
