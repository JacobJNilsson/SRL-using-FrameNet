from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement
import copy
from sklearn.feature_extraction import DictVectorizer


def dict_data(frames: List[Frame]) -> List[dict]:
    i = 0
    r = []
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
                        # For argument identification
                        # if role != "None":
                        #     role = 1
                        # else:
                        #     role = 0

                w = {
                    "lemma": " ".join(a.getLemma()),  # make the list a string
                    "pos": a.getPos(),
                    "deprel": a.getDeprel(),
                    "frame": frame_name,
                    "arg_role": role,
                }
                # subtrees = a.getSubtrees()
                r.append(w)
        i += 1
    return r


def split_data_to_identification_subsets(
    data: List,
    train_ratio: float = 0.5,
    test_ratio: float = 0.5,
    verification_ratio: float = 0,
) -> dict:
    # Leave the input unchanged
    data_copy = copy.deepcopy(data)

    # Change role to "boolean" one or zero
    dict_data_id = booleanize_role(data_copy)

    # Vectorize data
    vec = DictVectorizer()
    vector_data_id = vec.fit_transform(dict_data_id).toarray()

    # Split the test data
    x = [p[:-1] for p in vector_data_id]
    y = [p[-1] for p in vector_data_id]
    split = int(len(vector_data_id) * train_ratio)
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
) -> dict:
    # Leave the input unchanged
    data_copy = copy.deepcopy(data)

    # Remove entries without a role for role classification
    data_pruned = remove_none_role(data_copy)

    # Split classification data before vectorization
    x = []
    y = []
    for datapoint in data_pruned:
        xi = {}
        for key in datapoint:
            if key != "arg_role":
                xi[key] = datapoint[key]
            else:
                # y.append({key: datapoint[key]})
                y.append(datapoint[key])
        x.append(xi)
    assert len(y) == len(x)

    print(x[0])
    print(y[0])
    vec = DictVectorizer()
    x = vec.fit_transform(x).toarray()

    # Split the test data
    split = int(len(y) * train_ratio)
    r = {
        "x_train": x[:split],
        "x_test": x[split:],
        "y_train": y[:split],
        "y_test": y[split:],
    }

    # Vectorize data

    # y = vec.fit_transform(y).toarray()
    return r


def booleanize_role(data):

    # Change role to "boolean" one or zero
    for d in data:
        role = d["arg_role"]
        if role != "None":
            role = 1
        else:
            role = 0
        d["arg_role"] = role
    return data


def remove_none_role(data):
    r = []
    for d in data:
        if d["arg_role"] != "None":
            r.append(d)
    return r