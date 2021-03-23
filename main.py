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
from data_struct import Frame, TreeNode
from prune import pruneFrames
from format_data import (
    dict_data,
    split_data_to_identification_subsets,
    split_data_to_classification_subsets,
    sentence_data,
)
from training import cross_val, train_svm, test_svm, train_svm_2, cross_val
import time
import random


def main():
    print(time.localtime())
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
    # random.shuffle(sentence_data_raw)

    data_description = "Roles pruned where members < 6"

    # id_data = split_data_to_identification_subsets(dict_data_raw)
    cl_data = split_data_to_classification_subsets(
        train_ratio=1, test_ratio=0, data=dict_data_raw, description=data_description
    )

    last_time = timestamp(start, "Data parsing pruning and vectorizing: ")

    # Training identifier
    # id_clf = train_svm(
    #     x_data=id_data["x_train"],
    #     y_data=id_data["y_train"],
    #     prob=True,
    # )
    # last_time = timestamp(last_time, "Argument identification training:     ")

    # # Testing identifier
    # test_svm_2(
    #     id_clf,
    #     x_test=id_data["x_test"],
    #     y_test=id_data["y_test"],
    #     file="arg_ident_results.txt",
    #     display_conf_matrix=True,
    # )
    # last_time = timestamp(last_time, "Argument identification test:         ")

    # Training labeling
    # cl_clf = train_svm(
    #     x_data=cl_data["x_train"],
    #     y_data=cl_data["y_train"],
    # )
    # last_time = timestamp(last_time, "Argument classification training:     ")

    # # Testing labeling
    # test_svm(
    #     cl_clf,
    #     x_test=cl_data["x_train"],
    #     y_test=cl_data["y_train"],
    #     file="arg_class_results.txt",
    #     description=data_description,
    # )
    # last_time = timestamp(last_time, "Argument classification test:         ")

    # train_svm_2(
    #     x_train=cl_data["x_train"],
    #     y_train=cl_data["y_train"],
    #     x_test=cl_data["x_test"],
    #     y_test=cl_data["y_test"],
    # )
    # timestamp(last_time, "Argument classification training:     ")

    cross_val(x=cl_data["x_train"], y=cl_data["y_train"], description=data_description)
    last_time = timestamp(last_time, "Cross validation:                     ")
    timestamp(start, "Total time:                           ")


def timestamp(start: float, messege: str) -> float:
    end = time.time()
    t = end - start
    milisec = int((t % 1) * 1000)
    t = int(t)
    sec = t % 60
    min = (t % 3600) // 60
    hour = t // 3600
    print(
        messege
        + str(hour)
        + "h "
        + str(min)
        + "m "
        + str(sec)
        + "s "
        + str(milisec)
        + "ms"
    )
    return end


if __name__ == "__main__":
    main()
