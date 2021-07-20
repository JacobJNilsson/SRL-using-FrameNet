from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement
import copy

# Remove sentences without LU
def pruneFaltySentences(frames: List[Frame]) -> None:
    for f in frames:
        sentences = f.getSentences()
        for s in sentences:
            if s.getLU() == None:
                f.removeSentence(s)


def prune_sentences(sentences: List[Sentence]) -> List[TreeNode]:
    all_unpruned_words = []
    for sentence in sentences:
        unpruned_words = prune(sentence)
        all_unpruned_words.extend(unpruned_words)
    return all_unpruned_words


def pruneFromPredicate(predicates: List[TreeNode]) -> List[TreeNode]:
    remaining_words = []
    for p in predicates:
        while p:
            children = p.getSubtrees()
            parent = p.getParent()
            if parent and not parent in remaining_words and not parent in predicates:
                remaining_words.append(parent)
            for c in children:
                role = c.getPos()
                if (
                    role != "MID"
                    and role != "PAD"
                    and role != "MAD"
                    and role != "PUNCT"
                    and not c in predicates
                    and not c in remaining_words
                ):
                    remaining_words.append(c)
            p = parent

    return remaining_words


# Basic pruning of a sentence
def prune(sentence: Sentence) -> List[TreeNode]:
    fe_lu = sentence.getLU()
    if fe_lu == None:
        return []  # This should not be needed, but is :(
    lu_range = fe_lu.getRange()
    lus = []
    for i in range(lu_range[0], lu_range[1] + 1):
        lus += [sentence.getNode(i)]

    # Get the syntactic node

    remaining_words = pruneFromPredicate(lus)
    return remaining_words


def pruneSentencesInFrame(frame: Frame) -> None:
    sentences = frame.getSentences()
    for s in sentences:
        prune(s)


def pruneFrames(frames: List[Frame]) -> List[Frame]:
    # frames_copy = copy.deepcopy(frames)  #!! Does not work
    pruneFaltySentences(frames)
    for f in frames:
        pruneSentencesInFrame(f)
    return frames