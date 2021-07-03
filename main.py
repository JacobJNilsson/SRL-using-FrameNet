# from os import error
# from typing import List
from datetime import datetime
import os
from typing import List

from spacy.util import load_model
from data_parser import (
    parse,
    create_data,
    parse_syn_tree,
    parse_sem,
    parse_spacy,
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
from training import cross_val, train_svm, test_labeler, test_ident, cross_val
import time
import numpy as np
import sys
import random
import pickle

from message import send_email


def main():
    # Change this value to represent the data manipulation made
    data_description = "spacy test, all datapoints"

    start = time.time()
    now = datetime.now()
    dt_string = now.strftime("_%Y-%m-%d_%H-%M")
    directory = "run" + dt_string
    readable_time = now.strftime("%H:%M:%S %Y-%m-%d")
    print(f"Run started at: {readable_time}")
    send_email(
        data_description,
        f"New run started: {data_description} \nTime: {readable_time}\n",
        "jacobjnilsson@gmail.com",
    )

    ######## DATA READING AND MANIPULATION ########

    # Parse data
    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()
    res_frames = combineFrameLists(sem_frames, syn_frames)
    # expected_frames = combineFrameLists(sem_frames, syn_frames)

    spacy_frames = open_model("spacy_parse", ".")
    # spacy_frames = parse_spacy()
    # save_model(spacy_frames, "spacy_parse", ".")

    # print(spacy_frames[0].getSentences()[0])
    # print(spacy_frames[0].getSentences()[0].getRoot().getLemma())
    # print("\n\n")
    # print(res_frames[0].getSentences()[0])
    # timestamp(start, "Runtime: ")

    ###### MALTPARSER ######
    # # Prune data
    # #!!# I am not including the cases that are pruned, but are suposed to, in the tests!!!!
    # pruned_frames = pruneFrames(res_frames)

    # # Structure data in dicts
    # dict_data_raw = dict_data(pruned_frames)
    # # dict_data_all = expected_frames

    # # Just to make sure things are good and random (yet repeatable)
    # random.Random(1).shuffle(dict_data_raw)

    ##### SPACY #####
    # Prune data
    #!!# I am not including the cases that are pruned, but are suposed to, in the tests!!!!
    pruned_frames = pruneFrames(spacy_frames)

    # Structure data in dicts
    dict_data_raw = dict_data(pruned_frames)
    # dict_data_all = expected_frames

    # Just to make sure things are good and random (yet repeatable)
    random.Random(1).shuffle(dict_data_raw)

    id_data = split_data_to_identification_subsets(
        dict_data_raw, train_ratio=0.8, test_ratio=0.2
    )

    # cl_data = split_data_to_classification_subsets(
    #     train_ratio=0.8,
    #     test_ratio=0.2,
    #     data=dict_data_raw,
    #     description=data_description,
    # )

    last_time = timestamp(start, "Data parsing pruning and vectorizing: ")

    # C reate new run folder
    try:
        os.mkdir(directory)
    except:
        raise OSError("Unable to create directory")

    # Description of run
    f = open(directory + "/run_description.txt", "a")
    f.write(data_description)
    f.close()

    # ######## MODEL TRAINING ########

    send_email(
        data_description,
        f"Starting idendification training.\n",
        "jacobjnilsson@gmail.com",
    )

    # Training identifier
    id_clf = train_svm(
        x_data=id_data["x_train"],
        y_data=id_data["y_train"],
    )

    # Save model
    save_model(id_clf, "identification_model", directory)
    last_time = timestamp(last_time, "Argument identification training:     ")
    send_email(
        data_description,
        f"Identification svm trained! \n\n",
        "jacobjnilsson@gmail.com",
    )

    # Testing identifier
    ident_report = test_ident(
        id_clf,
        x_test=id_data["x_test"],
        y_test=id_data["y_test"],
        directory=directory,
        display_conf_matrix=True,
    )

    last_time = timestamp(last_time, "Argument identification test:         ")
    send_email(
        data_description,
        f"Identification stage done! \n\n {ident_report}",
        "jacobjnilsson@gmail.com",
    )

    #     # Training labeling
    #     cl_clf = train_svm(
    #         x_data=cl_data["x_train"],
    #         y_data=cl_data["y_train"],
    #         prob=True,
    #     )

    #     # Save model
    #     save_model(cl_clf, "classification_model", directory)
    #     last_time = timestamp(last_time, "Argument classification training:     ")

    #     # Run an existing model
    #     # cl_clf = open_model("classification_model", directory=directory)
    #     # last_time = timestamp(last_time, "Open model:                           ")

    #     # Testing labeling
    #     test_labeler(
    #         clf=cl_clf,
    #         x=cl_data["x_test"],
    #         y=cl_data["y_test"],
    #         directory=directory,
    #         frames=cl_data["f_test"],
    #         frame_data=res_frames,
    #     )
    #     last_time = timestamp(last_time, "Argument classification test:         ")

    # cross_val(
    #     x=cl_data["x_train"],
    #     y=cl_data["y_train"],
    #     frames=cl_data["f_train"],
    #     frame_data=res_frames,
    # )
    # last_time = timestamp(last_time, "Cross validation:                     ")

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


def save_model(clf, name, directory) -> None:
    with open(directory + "/" + name + ".pkl", "wb") as fid:
        pickle.dump(clf, fid)


def open_model(name, directory):
    with open(directory + "/" + name + ".pkl", "rb") as fid:
        return pickle.load(fid)


def save_to_file(l, name_of_file: str = "temp.txt"):
    f = open(name_of_file, "w")
    f.write(str(l))
    f.close()


if __name__ == "__main__":
    main()
