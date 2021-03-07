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
from argument_identification import argument_identification
import time


def main():
    start = time.time()
    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()
    res_frames = combineFrameLists(sem_frames, syn_frames)
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
        + milisec
        + "milisec"
    )
    vector_data = dict_data(res_frames)
    argument_identification(vector_data)
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
        + milisec
        + "milisec"
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
        + milisec
        + "milisec"
    )


if __name__ == "__main__":
    main()
