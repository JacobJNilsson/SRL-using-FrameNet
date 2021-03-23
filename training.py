from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import sys


def train_svm(
    x_data: List,
    y_data: List,
    gamma: float = 0.001,
    c: float = 100.0,
    kernel: str = "rbf",
    cache_size: int = 1000,
    prob: bool = False,
):

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma, C=c, cache_size=1000, kernel=kernel, probability=prob)

    # Learn the data on the train subset
    clf.fit(x_data, y_data)

    return clf


def train_svm_2(
    x_train: List, y_train: List, x_test: List, y_test: List, kernel: str = "rbf"
):
    # Set the parameters by cross-validation
    tuned_parameters = [
        # {
        #     "kernel": ["rbf"],
        #     "gamma": [1e-3, 1e-4],
        #     "C": [1, 10, 100, 1000],
        #     "cache_size": [2000],
        #     "zero_division": [1],
        # },
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000],
            "cache_size": [2000],
        },
    ]

    scores = ["precision", "recall"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring="%s_macro" % score)
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred, zero_division=1))
        print()


def test_svm(
    clf,
    x_test: List,
    y_test: List,
    file: str = None,
    display_conf_matrix: bool = False,
    description: str = "",
):

    # Predict the value of the digit on the test subset
    predicted = clf.predict(x_test)

    print(
        f"Classification report for classifier {clf}{description}:\n"
        f"{metrics.classification_report(y_test, predicted, zero_division=1)}\n"
    )

    if file:
        f = open(file, "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        if display_conf_matrix:
            # This takes a crazy long time, why?
            disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
            disp.figure_.suptitle("Confusion Matrix")
            print(f"Confusion matrix:\n{disp.confusion_matrix}")

            f.write(f"{dt_string}:\n{disp.confusion_matrix}\n\n")
            f.close()
        else:
            f.write(
                f"{dt_string}:\nClassification report for classifier {clf}{description}:\n"
            )
            f.write(
                f"{metrics.classification_report(y_test, predicted, zero_division=1)}\n"
            )
            f.close()


def cross_val(
    x: List,
    y: List,
    file: str = "cross_validation.txt",
    description: str = "",
    gamma: float = 0.001,
    c: float = 100.0,
    kernel: str = "rbf",
    cache_size: int = 1000,
    prob: bool = True,
):
    f = open(file, "a")
    f.write(description + "\n")
    f.write(str(np.unique(y)))
    clf = svm.SVC(
        gamma=gamma, C=c, cache_size=cache_size, kernel=kernel, probability=prob
    )
    proba = cross_val_predict(
        clf,
        x,
        y,
        method="predict_proba",
        n_jobs=4,
    )
    np.set_printoptions(threshold=sys.maxsize)
    f.write(str(proba))
