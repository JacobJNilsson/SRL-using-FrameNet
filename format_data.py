from datetime import datetime
from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement
import copy
from sklearn.feature_extraction import DictVectorizer
import ast


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
                    head_name = (parent.getWord(),)
                    head_lemma = (parent.getLemma(),)
                    head_pos = (parent.getPos(),)
                    head_deprel = (parent.getDeprel(),)
                else:
                    head_name = None
                    head_lemma = None
                    head_pos = None
                    head_deprel = None

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
                    "ce": f.getCoreElements(),
                    "pe": f.getPeripheralElements(),
                    "child_word": child_word,
                    "child_lemma": child_lemma,
                    "child_pos": child_pos,
                    "child_deprel": child_deprel,
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


def split_data_to_identification_subsets(
    data: List,
    train_ratio: float = 0.5,
    test_ratio: float = 0.5,
    verification_ratio: float = 0,
) -> dict:
    # Leave the input unchanged
    #!! DOES NOT WORK
    # data_copy = copy.deepcopy(data)

    no_datapoints = -1

    # Change role to "boolean" one or zero
    bool_data = booleanize_role(data)
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
    data_copy = copy.deepcopy(data)

    # Filter data
    filter_list = []
    f = open("class_occurance.txt", "r", encoding="utf8")
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
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    classes = {i: y.count(i) for i in y}
    f = open("class_occurance.txt", "a", encoding="utf8")
    f.write(dt_string + ":\n")
    f.write(description + "\n")
    f.write(str(classes) + "\n\n")
    i = 0
    sum = 0
    max = 0
    maxClass = ""
    min = 999999
    minClass = ""
    for c in classes:
        i += 1
        val = classes[c]
        sum += val
        if max < val:
            max = val
            maxClass = c
        if min > val:
            min = val
            minClass = c
    f.write("No. classes total: " + str(i) + "\n")
    f.write("No. members total: " + str(sum) + "\n")
    f.write("Averege no. members per class: " + str(sum / i) + "\n")
    f.write("Class with max no. members: " + maxClass + ", " + str(max) + "\n")
    f.write("Class with min no. members: " + minClass + ", " + str(min) + "\n\n\n")
    f.close()

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
        # print(role)
        if role != "None" and role != None:
            role = 1.0
        else:
            role = 0.0
        # print(f"got role {role}\n")
        d["arg_role"] = role
    return data


def filter_roles(data: List[dict], filter_list: list):
    r = []
    for d in data:
        role = d["arg_role"]
        if role != "None" and not role in filter_list and not role == "LU":
            r.append(d)
    return r


def filter_roles_in_sentences(data: List[List[dict]], filter_list: list):
    r = []
    for s in data:
        sentence = []
        for d in s:
            role = d["arg_role"]
            if role != "None" and not role in filter_list:
                sentence.append(d)
        if sentence != []:
            r.append(sentence)
    return r