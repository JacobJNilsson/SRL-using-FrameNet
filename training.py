from format_data import Frame
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import sys
import os
from joblib import parallel_backend


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
    clf = svm.SVC(
        gamma=gamma,
        C=c,
        cache_size=cache_size,
        kernel=kernel,
        probability=prob,
        verbose=True,
    )

    # Learn the data on the train subset
    with parallel_backend("threading", n_jobs=-1):
        clf.fit(x_data, y_data)

    return clf


def grid_search(
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


def test_ident(
    clf,
    x_test: List,
    y_test: List,
    directory: str = None,
    display_conf_matrix: bool = True,
    description: str = "",
):

    # Predict the value of the digit on the test subset
    with parallel_backend("threading", n_jobs=-1):
        predicted = clf.predict(x_test)
    return_string = f"Classification report for classifier {clf}{description}:\n {metrics.classification_report(y_test, predicted, zero_division=1)}\n"
    print(return_string)

    if display_conf_matrix:
        # This takes a crazy long time, why?
        disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        matrix_string = f"Confusion matrix:\n{disp.confusion_matrix}\n"
        print(matrix_string)
        return_string += "\n" + matrix_string

    if directory:
        f = open(directory + "/identifier_metrics.txt", "a")
        f.write(return_string)
        f.close()
    return return_string


def test_labeler(
    x: List,
    y: List[str],
    clf,
    directory: str,
    frames: List,
    frame_data: List[Frame],
):

    probability_matrix = clf.predict_proba(x)
    print("Done calculating probability")

    role_key = np.unique(y)

    np.set_printoptions(threshold=sys.maxsize)

    # save the role key
    f0 = open(directory + "/labeling_key.txt", "w")
    f0.write(str(role_key) + "\n")
    f0.close()

    # save the probabilities
    f1 = open(directory + "/labeling_raw_predictions.txt", "w")
    f1.write(str(role_key) + "\n")
    f1.write(str(probability_matrix))
    f1.close()

    # !! not needed?
    # save the frames of the input data
    f2 = open(directory + "/labeling_input_frames.txt", "w")
    f2.write(str(frames))
    f2.close()

    # save the expected output
    f3 = open(directory + "/labeling_expected_output.txt", "w")
    f3.write(str(y))

    prediction = calc_proba(
        frame_data=frame_data,
        input_frames=frames,
        role_key=role_key,
        directory=directory,
        proba_matrix=probability_matrix,
    )

    # save the predicted output
    f3 = open(directory + "/labeling_predictions.txt", "w")
    f3.write(str(prediction))
    f3.close()

    # save the classification report
    m = metrics.classification_report(y, prediction, zero_division=1)
    f4 = open(directory + "/labeling_metrics.txt", "w")
    f4.write(f"{m}\n")
    f4.close()


#!! needs more work before use (see test_labeler())
def cross_val(
    x: List,
    y: List,
    frames: List,
    frame_data: List[Frame],
    file: str = "cross_validation",
    description: str = "",
    gamma: float = 0.001,
    c: float = 100.0,
    kernel: str = "rbf",
    cache_size: int = 1000,
    prob: bool = True,
    clf=None,
):
    now = datetime.now()
    dt_string = now.strftime("_%d-%m-%Y %H:%M")
    result_folder = "results" + dt_string + "/"
    if clf == None:
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
    print("Done with cross validation")
    role_key = np.unique(y)

    np.set_printoptions(threshold=sys.maxsize)

    f1 = open(result_folder + file + dt_string + ".txt", "a")
    f1.write(description + "\n")
    f1.write(str(role_key) + "\n")
    f1.write(str(proba))
    f1.close()

    f2 = open(result_folder + "input_frames" + dt_string + ".txt", "a")
    f2.write(description + "\n")
    f2.write(str(frames))
    f2.close()

    f3 = open(result_folder + "output" + dt_string + ".txt", "a")
    f3.write(description + "\n")
    f3.write(str(y))

    prediction = calc_proba(
        frame_data=frame_data,
        input_frames=frames,
        role_key=role_key,
    )

    f3 = open(result_folder + "result" + dt_string + ".txt", "a")
    f3.write(description + "\n")
    f3.write(prediction)
    f3.close()


def calc_proba(
    frame_data: List[Frame],
    input_frames=None,
    role_key=None,
    proba_matrix=None,
    directory=None,
) -> List[str]:
    # if not input_frames and directory:
    #     # read input_frames
    #     # with open(directory + "/") as fp:
    #     #     data = [ast.literal_eval(line) for line in fp if line.strip()]
    #     a = None
    # if not output and dir:
    #     # read output
    #     a = None
    # if not proba_matrix and directory:
    #     # read matrix
    #     with open(directory + "/labeling_raw_predictions.txt") as fp:
    #         text = fp.read().strip("\n")
    #         proba_matrix = eval(text)
    # if role_key == None and directory:
    #     # read role if role_key
    #     a = None

    pruned_proba_matrix = []
    for i in range(len(proba_matrix)):
        p = proba_matrix[i]
        frame = input_frames[i]
        pruned_proba_matrix.append(prune_roles_to_frame(frame, role_key, p, frame_data))

    # Get the actual predictions
    # predictions = predict(pruned_proba_matrix)
    predictions = predict(proba_matrix)

    return predictions


def prune_roles_to_frame(
    frame: str, role_key: List[str], proba_array: List[float], frame_data: List[Frame]
) -> dict:
    r = {}
    target_frame = None
    for f in frame_data:
        if frame.strip() == f.getName().strip():
            target_frame = f
            break
    if not target_frame:
        print("Frame is in list: " + str(frame in [f.getName() for f in frame_data]))
        raise NameError("Frame in not found: '" + frame + "'")

    roles = target_frame.getCoreElements() + target_frame.getPeripheralElements()
    assert roles != []

    for i in range(len(role_key)):
        role = role_key[i]
        if role in roles:
            r[role] = proba_array[i]

    assert r != {}

    return r


def predict(proba_matrix) -> List[str]:
    predictions = []
    for row in proba_matrix:
        prediction = None
        prediction_value = 0
        for key in row:
            if row[key] > prediction_value:
                prediction = key
                prediction_value = row[key]
        predictions.append(prediction)
    return predictions
