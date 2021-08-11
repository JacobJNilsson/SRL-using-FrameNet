from data_struct import Sentence, TreeNode
from format_data import Frame, create_result_data, create_result_data
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics, calibration
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
import sys
import os
from joblib import parallel_backend
from scipy.sparse import csr_matrix
# from helpers import *


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
    clf = svm.LinearSVC(
        C=c,
        random_state=1,
        max_iter=100000,
    )
    if prob:
        clf = calibration.CalibratedClassifierCV(clf)

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

        clf = GridSearchCV(svm.SVC(), tuned_parameters,
                           scoring="%s_macro" % score)
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
    f3.close()

    # prediction = calc_proba(
    #     frame_data=frame_data,
    #     input_frames=frames,
    #     role_key=role_key,
    #     directory=directory,
    #     proba_matrix=probability_matrix,
    # )

    # # save the predicted output
    # f3 = open(directory + "/labeling_predictions.txt", "w")
    # f3.write(str(prediction))
    # f3.close()

    # # save the classification report
    # m = metrics.classification_report(y, prediction, zero_division=1)
    # f4 = open(directory + "/labeling_metrics.txt", "w")
    # f4.write(f"{m}\n")
    # f4.close()


def cross_val(
    directory: str,
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

    result_folder = directory + "/results"
    if clf == None:
        clf = calibration.CalibratedClassifierCV(
            svm.LinearSVC(
                C=c,
                verbose=True,
                random_state=1,
                max_iter=100000,
            )
        )
    probabilities = cross_val_predict(
        clf,
        x,
        y,
        method="predict_proba",
        n_jobs=4,
    )
    print("Done with cross validation")
    classes = np.unique(y)
    np.set_printoptions(threshold=sys.maxsize)
    try:
        os.mkdir(result_folder)
    except:
        raise OSError("Unable to create directory")

    f1 = open(result_folder + "/" + file + ".txt", "a")
    f1.write(str(classes) + "/n")
    f1.write(str(probabilities))
    f1.close()

    f2 = open(result_folder + "/output" + ".txt", "a")
    f2.write(str(y))

    predictions = calc_proba(classes, probabilities)

    f3 = open(result_folder + "/result" + ".txt", "a")
    f3.write(str(predictions))
    f3.close()

    # save the classification report
    m = metrics.classification_report(y, predictions, zero_division=1)
    f4 = open(directory + "/labeling_metrics.txt", "w")
    f4.write(f"{m}\n")
    f4.close()
    print(f"{m}")
    return clf


def calc_predictions(classes, probabilities) -> List[str]:
    r = []
    for p in probabilities:
        best_guess_value = 0
        best_guess = None
        for c, probability in zip(classes, p):
            if probability > best_guess_value:
                best_guess = c
                best_guess_value = probability
        r.append(best_guess)
    return r


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
        print("Frame is in list: " +
              str(frame in [f.getName() for f in frame_data]))
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
        for key in range(len(row)):
            if row[key] > prediction_value:
                prediction = key
                prediction_value = row[key]
        predictions.append(prediction)
    return predictions


def train_classifier(
    words: List[TreeNode],
    bool_result: bool,
    c: float = 100.0,
    prob: bool = False,
):

    # Extract feature data
    X_data = csr_matrix([w.getFeatures() for w in words])
    y_data = create_result_data(words, bool_result)
    print(f"{y_data[:20]=}")
    # assert(len(X_data) == len(y_data))
    assert(X_data.shape[0] == len(y_data))

    # Check the distribution of y_data
    occurance_class, max_class, avg = dist_max_avg(y_data)
    ratio = occurance_class[max_class]/avg
    ratio_string = f"Ratio between the most occuring and the avg: {occurance_class[max_class]/avg}"
    if bool_result:
        report = f"Report of the y_data distribution\nThe different classes occurances: {occurance_class}\n{ratio_string}"
    else:
        report = f"Report of the y_data distribution\nThe most occuring class: {max_class} {occurance_class[max_class]}\n{ratio_string}"
    print(report)
    # Create a classifier: a support vector classifier
    clf = svm.LinearSVC(
        C=c,
        verbose=False,
        random_state=1,
        max_iter=100000,
    )
    if prob:
        clf = calibration.CalibratedClassifierCV(clf)

    # Learn the data on the train subset
    with parallel_backend("threading", n_jobs=-1):
        clf.fit(X_data, y_data)

    return clf, report


def test_classifier(
    clf: svm.LinearSVC,
    words: List[TreeNode],
    bool_result: bool,
) -> str:

    # Extract feature data
    X_data = [w.getFeatures() for w in words]
    y_data = create_result_data(words, bool_result)
    assert(len(X_data) == len(y_data))

    if bool_result:
        # print(f"testing identifier with {len(X_data)} datapoints")
        with parallel_backend("threading", n_jobs=-1):
            predictions = clf.predict(X_data)
    else:
        # print(f"testing labeler with {len(X_data)} datapoints")
        probabilities = clf.predict_proba(X_data)
        # !Probably something wrong here!
        predictions = calc_predictions(clf.classes_, probabilities)

    # Add the prediction to the word objects
    for word, prediction in zip(words, predictions):
        word.addPrediction(prediction)
        if not bool_result:
            word.correctChildren(prediction)
    # Classifier evaluation
    evaluation = metrics.classification_report(
        y_data, predictions, zero_division=1)
    return evaluation


def evaluate_sentences(sentences: List[Sentence]):
    y = []
    p = []
    for s in sentences:
        roles, predictions = s.getRolesAndPredictions()
        y.extend(roles)
        p.extend(predictions)
    evaluation = metrics.classification_report(y, p, zero_division=1)
    return evaluation


def dist_max_avg(classes: list):
    occurance_class = {}
    for c in classes:
        if c in occurance_class.keys():
            occurance_class[c] += 1
        else:
            occurance_class[c] = 1
    max_class = None
    sum_max = 0
    sum_all = 0
    for k, sum_k in occurance_class.items():
        sum_k = occurance_class[k]
        sum_all += sum_k
        if sum_k > sum_max:
            sum_max = sum_k
            max_class = k
    avg = sum_all/len(occurance_class.keys())

    return occurance_class, max_class, avg
