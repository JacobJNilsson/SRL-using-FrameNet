# from os import error
# from typing import List
# import data_struct as DS
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
from format_data import vectorize_data


def main():

    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()
    res_frames = combineFrameLists(sem_frames, syn_frames)

    pruneFrames(res_frames)
    vector_data = vectorize_data(res_frames)
    print()
    print(vector_data[0:20])


if __name__ == "__main__":
    main()
