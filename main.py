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
from format_data import dict_data
from training import train_svm
import time
import copy


def main():
    start = time.time()

    # Parse data
    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()
    res_frames = combineFrameLists(sem_frames, syn_frames)

    # Prune data
    pruneFrames(res_frames)
    mid = time.time()
    t = mid - start
    milisec = int((t % 1) * 1000)
    t = int(t)
    print(
        "Data parsing and pruning:    "
        + str(t // 60)
        + "m "
        + str(t % 60)
        + "s "
        + str(milisec)
        + "ms"
    )
    vector_data_raw = dict_data(res_frames)
    vector_data_id = copy.deepcopy(vector_data_raw)
    for d in vector_data_raw:
        role = d["arg_role"]
        if role != "None":
            role = 1
        else:
            role = 0
        d["arg_role"] = role

    train_svm(vector_data_id, file="arg_ident_results.txt")
    end = time.time()
    t = end - mid
    milisec = int((t % 1) * 1000)
    t = int(t)
    print(
        "Argument identification:     "
        + str(t // 60)
        + "m "
        + str(t % 60)
        + "s "
        + str(milisec)
        + "ms"
    )
    t = end - start
    milisec = int((t % 1) * 1000)
    t = int(t)
    print(
        "Total time:                  "
        + str(t // 60)
        + "m "
        + str(t % 60)
        + "s "
        + str(milisec)
        + "ms"
    )


if __name__ == "__main__":
    main()
