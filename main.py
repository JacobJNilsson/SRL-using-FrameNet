# from os import error
# from typing import List
from data_parser import (
    parse,
    create_data,
    parse_syn_tree,
    parse_sem,
    getFrame,
    compareFrames,
    combineFrameLists,
)
from data_struct import Frame
from prune import pruneFrames
from format_data import (
    dict_data,
    split_data_to_identification_subsets,
    split_data_to_classification_subsets,
)
from training import train_svm, test_svm
import time
import random


def main():
    start = time.time()

    # Parse data
    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()
    res_frames = combineFrameLists(sem_frames, syn_frames)

    # Prune data
    pruneFrames(res_frames)

    # Structure data in dicts
    dict_data_raw = dict_data(res_frames)

    # Just to make sure things are good and random
    random.shuffle(dict_data_raw)

    id_data = split_data_to_identification_subsets(dict_data_raw)
    cl_data = split_data_to_classification_subsets(dict_data_raw)

    last_time = timestamp(start, "Data parsing pruning and vectorizing: ")

    id_clf = train_svm(
        x_data=id_data["x_train"],
        y_data=id_data["y_train"],
    )

    last_time = timestamp(last_time, "Argument identification training:     ")

    test_svm(
        id_clf,
        x_test=id_data["x_test"],
        y_test=id_data["y_test"],
        file="arg_ident_results.txt",
        display_conf_matrix=True,
    )

    last_time = timestamp(last_time, "Argument identification test:         ")

    cl_clf = train_svm(
        x_data=cl_data["x_train"],
        y_data=cl_data["y_train"],
    )

    last_time = timestamp(last_time, "Argument classification training:     ")

    test_svm(
        cl_clf,
        x_test=cl_data["x_test"],
        y_test=cl_data["y_test"],
        file="arg_class_results.txt",
    )

    last_time = timestamp(last_time, "Argument classification test:         ")

    timestamp(start, "Total time:                           ")


def timestamp(start: float, messege: str) -> float:
    end = time.time()
    t = end - start
    milisec = int((t % 1) * 1000)
    t = int(t)
    print(messege + str(t // 60) + "m " + str(t % 60) + "s " + str(milisec) + "ms")
    return end


if __name__ == "__main__":
    main()
