from datetime import datetime
from math import floor
from typing import List

from pandas.core import frame
from data_struct import Frame, Sentence, TreeNode, FrameElement
import copy
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import ast
import random


def dict_data(frames: List[Frame]) -> List[dict]:
    i = 0
    r = []
    # print("Creating dict from data")
    for f in frames:
        frame_name = f.getName()
        for s in f.getSentences():
            arguments = s.getArguments()
            frame_elements = s.getFrameElements()
            for a in arguments:
                role = None
                ref = a.getRef()
                for fe in frame_elements:
                    fe_range = fe.getRange()
                    if ref > fe_range[0] and ref <= fe_range[1] + 1:
                        role = fe.getName()
                parent = a.getParent()
                if parent != None:
                    head_name = parent.getWord()
                    head_lemma = parent.getLemma()
                    head_pos = parent.getPos()
                    head_deprel = parent.getDeprel()
                else:
                    head_name = "None"
                    head_lemma = "None"
                    head_pos = "None"
                    head_deprel = "None"

                child_word = []
                child_lemma = []
                child_pos = []
                child_deprel = []
                for child in a.getSubtrees():
                    child_word.append(child.getWord())
                    child_lemma.append(child.getLemma())
                    child_pos.append(child.getPos())
                    child_deprel.append(child.getDeprel())

                w = {
                    "word": a.getWord(),
                    "lemma": a.getLemma(),
                    "pos": a.getPos(),
                    "deprel": a.getDeprel(),
                    "frame": frame_name,
                    "head_name": head_name,
                    "head_lemma": head_lemma,
                    "head_pos": head_pos,
                    "head_deprel": head_deprel,
                    # "ce": f.getCoreElements(), # listor
                    # "pe": f.getPeripheralElements(), # listor
                    # "child_word": child_word, # listor
                    # "child_lemma": child_lemma, # listor
                    # "child_pos": child_pos, # listor
                    # "child_deprel": child_deprel, # listor
                    "arg_role": role,  # Classification data (result)
                }
                # print(f"{a.getLemma()=}")
                # subtrees = a.getSubtrees()
                r.append(w)
        i += 1
    return r


def sentence_data(frames: List[Frame]) -> List[List[dict]]:
    i = 0
    r = []
    for f in frames:
        frame_name = f.getName()
        for s in f.getSentences():
            sentence = []
            arguments = s.getArguments()
            frame_elements = s.getFrameElements()
            for a in arguments:
                role = None
                ref = a.getRef()
                for fe in frame_elements:
                    fe_range = fe.getRange()
                    if ref > fe_range[0] and ref <= fe_range[1] + 1:
                        role = fe.getName()
                w = {
                    "lemma": " ".join(a.getLemma()),  # make the list a string
                    "pos": a.getPos(),
                    "deprel": a.getDeprel(),
                    "frame": frame_name,
                    "arg_role": role,
                }
                # This will be needed in the future
                # subtrees = a.getSubtrees()
                sentence.append(w)
            r.append(sentence)
        i += 1
    return r


def split_data_train_test(
    frames: List[Frame], train_ratio=0.8, random_state=0
) -> tuple:
    all_sentences = []
    for frame in frames:
        sentences = frame.getSentences()
        all_sentences.extend(sentences)
    random.Random(1).shuffle(all_sentences)
    split = int(len(all_sentences) * train_ratio)
    # TODO Make sure there is no instance of a test sentence that contain roles not included in the training set

    train_sentences = all_sentences[:split]
    test_sentences = all_sentences[split:]
    return (train_sentences, test_sentences)


def split_data_to_identification_subsets(
    data: List,
    train_ratio: float = 0.5,
    test_ratio: float = 0.5,
    verification_ratio: float = 0,
) -> dict:
    # Leave the input unchanged
    #!! DOES NOT WORK
    data_copy = copy.deepcopy(data)
    # data_copy = data

    no_datapoints = -1

    # Change role to "boolean" one or zero
    bool_data = booleanize_role(data_copy)
    role_id_list = [d.pop("arg_role") for d in bool_data][0:no_datapoints]
    feature_id_dict = bool_data[0:no_datapoints]
    print(f"Number of datapoints {len(role_id_list)}")

    # Vectorize data
    vec = DictVectorizer()
    vector_feature_id = vec.fit_transform(feature_id_dict).toarray()

    # Split the test data
    x = vector_feature_id
    y = role_id_list
    assert len(y) == len(x)

    split = int(len(y) * train_ratio)
    r = {
        "x_train": x[:split],
        "x_test": x[split:],
        "y_train": y[:split],
        "y_test": y[split:],
    }
    return r


def split_data_to_classification_subsets(
    data: List,
    train_ratio: float = 0.5,
    test_ratio: float = 0.5,
    verification_ratio: float = 0,
    description="",
) -> dict:
    # Leave the input unchanged
    #!! DOES NOT WORK
    data_copy = copy.deepcopy(data)
    # data_copy = data

    # Filter data
    filter_list = []
    f = open("class_occurrence.txt", "r", encoding="utf8")
    first_line = f.readline()
    classes = ast.literal_eval(first_line)
    for key in classes:
        # !! this is the expression for filtering
        if classes[key] < 6:
            filter_list.append(key)

    # Remove entries without a role for role classification
    # (Assume that role identification is perfect)
    data_filtered = filter_roles(data_copy, filter_list)

    # Split classification data before vectorization
    x = []
    y = []
    for datapoint in data_filtered:
        xi = {}
        for key in datapoint:
            if key != "arg_role":
                xi[key] = datapoint[key]
            else:
                y.append(datapoint[key])
        x.append(xi)
    assert len(y) == len(x)

    # Saving the class information for analysis
    # now = datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # classes = {i: y.count(i) for i in y}
    # f = open("class_occurrence.txt", "a", encoding="utf8")
    # f.write(dt_string + ":\n")
    # f.write(description + "\n")
    # f.write(str(classes) + "\n\n")
    # i = 0
    # sum = 0
    # max = 0
    # maxClass = ""
    # min = 999999
    # minClass = ""
    # for c in classes:
    #     i += 1
    #     val = classes[c]
    #     sum += val
    #     if max < val:
    #         max = val
    #         maxClass = c
    #     if min > val:
    #         min = val
    #         minClass = c
    # f.write("No. classes total: " + str(i) + "\n")
    # f.write("No. members total: " + str(sum) + "\n")
    # f.write("Averege no. members per class: " + str(sum / i) + "\n")
    # f.write("Class with max no. members: " + maxClass + ", " + str(max) + "\n")
    # f.write("Class with min no. members: " + minClass + ", " + str(min) + "\n\n\n")
    # f.close()

    # Get a list of frames
    f = [p["frame"] for p in x]
    assert len(x) == len(f)

    vec = DictVectorizer()
    x = vec.fit_transform(x).toarray()

    # Split the test data
    split = int(len(y) * train_ratio)
    r = {
        "x_train": x[:split],
        "x_test": x[split:],
        "y_train": y[:split],
        "y_test": y[split:],
        "f_train": f[:split],
        "f_test": f[split:],
    }

    return r


def booleanize_role(data):

    # Change role to "boolean" one or zero
    for d in data:
        role = d["arg_role"]
        if role != "None" and role != None:
            role = 1.0
        else:
            role = 0.0
        d["arg_role"] = role
    return data


def create_result_data(words, bool_result):
    y_data = [w.getRole() for w in words]
    if bool_result:
        y_data = [0 if role == "None" else 1 for role in y_data]
    return y_data


def create_feature_data(words: List[TreeNode], features):
    # create feature data
    X_data = []
    for word in words:
        w = {}
        lus = word.getLUs()
        # print(f"{[lu.getWord() for lu in lus]}\n")
        head = word.getParent()
        children = word.getSubtrees()
        if "frame" in features:
            w["frame"] = word.getFrame().getName()
        if "core_elements" in features:
            w["core_elements"] = word.getFrame().getCoreElements()
        if "word" in features:
            w["word"] = word.getWord()
        if "lemma" in features:
            w["lemma"] = word.getLemma()
        if "pos" in features:
            w["pos"] = word.getPos()
        if "deprel" in features:
            w["deprel"] = word.getDeprel()
        if "ref" in features:
            w["ref"] = word.getRef()
        if "lu_words" in features:
            lu_words = []
            for lu in lus:
                lu_words.extend(lu.getWord())
            w["lu_words"] = lu_words
        if "lu_lemmas" in features:
            lu_lemmas = []
            for lu in lus:
                lu_lemmas.extend(lu.getLemma())
            w["lu_lemmas"] = lu_lemmas
        if "lu_deprels" in features:
            lu_deprels = []
            for lu in lus:
                lu_deprels.append(lu.getDeprel())
            w["lu_deprels"] = lu_deprels
        if "lu_pos" in features:
            lu_pos = []
            for lu in lus:
                lu_pos.append(lu.getPos())
            w["lu_pos"] = lu_pos
        if "head_word" in features:
            if head != None:
                head_word = head.getWord()
            else:
                head_word = "None"
            w["head_word"] = head_word
        if "head_lemma" in features:
            if head != None:
                head_lemma = head.getLemma()
            else:
                head_lemma = "None"
            w["head_lemma"] = head_lemma
        if "head_deprel" in features:
            if head != None:
                head_deprel = head.getDeprel()
            else:
                head_deprel = None
            w["head_deprel"] = head_deprel
        if "head_pos" in features:
            if head != None:
                head_pos = head.getPos()
            else:
                head_pos = "None"
            w["head_pos"] = head_pos
        if "child_words" in features:
            child_words = []
            for child in children:
                child_words.append(child.getWord())
            w["child_words"] = child_words
        if "child_lemmas" in features:
            child_lemmas = []
            for child in children:
                child_lemmas.extend(child.getLemma())
            w["child_lemmas"] = child_lemmas
        if "child_deprels" in features:
            child_deprels = []
            for child in children:
                child_deprels.append(child.getDeprel())
            w["child_deprels"] = child_deprels
        if "child_pos" in features:
            child_pos = []
            for child in children:
                child_pos.append(child.getPos())
            w["child_pos"] = child_pos
        X_data.append(w)

    # The dtype is set to np.bool_ since all features are 1 or 0
    # This should change if the child features started counting occurances.
    vec = DictVectorizer(dtype=np.bool_)
    X_data = np.array(vec.fit_transform(X_data).toarray())
    return X_data


def create_feature_representation(frames: List[Frame], extract_features):
    words: List[TreeNode] = []
    for f in frames:
        for s in f.getSentences():
            for w in s.getTreeNodesOrdered():
                words.append(w)
    features = create_feature_data(words, extract_features)
    for w, f in zip(words, features):
        # The actual change this function does
        w.addFeatures(f)
    # Returns a string for logging purposes
    return f"Number of data points: {len(features)}\nNumber of features: {len(features[0])}"
