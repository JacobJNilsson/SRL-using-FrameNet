from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement


def vectorize_data(frames: List[Frame]) -> List[List]:
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
                w = [a.getLemma(), a.getPos(), a.getDeprel(), frame_name, role]
                # subtrees = a.getSubtrees()
                r.append(w)
        i += 1
    return r
