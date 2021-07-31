from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement
import copy

# Different filters for different stages in the pipeline.
# The different filters have very chaotic return protocols.


# Remove sentences without LU
def filter_faulty_sentences(frames: List[Frame]) -> None:

    for f in frames:
        sentences = f.getSentences()
        for s in sentences:
            if not s.getLUs():
                f.removeSentence(s)


# Prunes the sentences and returns the words chosen for data set
def prune_sentences(sentences: List[Sentence], filter: dict) -> List[TreeNode]:
    r_words = []
    for sentence in sentences:
        if "prune" in filter.keys():
            prune_method = filter["prune"]
            if prune_method == 0:
                unpruned_words = prune(sentence)
                r_words.extend(unpruned_words)
            elif prune_method == 1:
                fes = sentence.getFrameElements()
                for fe in fes:
                    words_with_role = sentence.getNodesInFrameElement(fe)
                    r_words.extend(words_with_role)
                    lus = []
                    if fe.getName() == "LU":
                        lus.extend(words_with_role)
                    words_without_role = prune_from_predicates(
                        lus, no_role=True)
                    r_words.extend(words_without_role)
            else:
                r_words.extend(sentence.getTreeNodesOrdered())
        else:
            r_words.extend(sentence.getTreeNodesOrdered())
    return r_words


# Basic pruning of a sentence
def prune(sentence: Sentence) -> List[TreeNode]:
    fe_lus = sentence.getLUs()
    lus = []
    for fe_lu in fe_lus:
        lus.extend(sentence.getNodesInFrameElement(fe_lu))

    remaining_words = prune_from_predicates(lus)
    return remaining_words


def prune_from_predicates(predicates: List[TreeNode], no_role=False) -> List[TreeNode]:
    remaining_words = set()
    for w in predicates:
        while w:
            if not w in remaining_words:
                if not no_role:
                    remaining_words.add(w)
                elif w.getRole() == "None":
                    remaining_words.add(w)

                children = w.getSubtrees()
                for child in children:
                    role = child.getPos()
                    if (
                        role != "MID"
                        and role != "PAD"
                        and role != "MAD"
                        and role != "PUNCT"
                        and not child in predicates
                        and not child in remaining_words
                    ):
                        if not no_role:
                            remaining_words.add(child)
                        elif child.getRole() == "None":
                            remaining_words.add(child)
                w = w.getParent()
            else:
                w = None
    remaining_words = list(remaining_words)
    if no_role:
        for w in remaining_words:
            assert w.getRole() == "None"
    return list(remaining_words)


def prune_sentences_in_frame(frame: Frame) -> None:
    sentences = frame.getSentences()
    for s in sentences:
        prune(s)


def prune_frames(frames: List[Frame]) -> List[Frame]:
    filter_faulty_sentences(frames)
    for f in frames:
        prune_sentences_in_frame(f)
    return frames


def filter_data(frames: List[Frame], filter: dict):
    (frames, no_filtered_sentences) = filter_sentences(frames, filter)
    (frames, no_filtered_frames) = filter_frames(frames, filter)
    return (frames, f"Number of filtered frames: {no_filtered_frames}\nNumber of filtered sentences: {no_filtered_sentences}")


def filter_sentences(frames: List[Frame], filter: dict):
    role_occurance = {}
    no_filtered_sentences = 0
    for frame in frames:
        sentences = frame.getSentences()
        # prune sentences depending on role occurance
        for sentence in sentences:
            fes = sentence.getFrameElements()
            for fe in fes:
                role = fe.getName()
                if role in role_occurance.keys():
                    role_occurance[role] += 1
                else:
                    role_occurance[role] = 1
    if filter.__contains__("min_role_occurance"):
        for frame in frames:
            sentences = frame.getSentences()
            for sentence in sentences:
                fes = sentence.getFrameElements()
                for fe in fes:
                    role = fe.getName()
                    if role_occurance[role] < filter["min_role_occurance"]:
                        frame.removeSentence(sentence)
                        no_filtered_sentences += 1
    if filter.__contains__("max_role_occurance"):
        for frame in frames:
            sentences = frame.getSentences()
            for sentence in sentences:
                fes = sentence.getFrameElements()
                for fe in fes:
                    if role_occurance[fe.getName()] > filter["max_role_occurance"]:
                        frame.removeSentence(sentence)
                        no_filtered_sentences += 1
    return (frames, no_filtered_sentences)


def filter_frames(frames: List[Frame], filter: dict):
    r_frames = []
    no_filtered_frames = 0
    for frame in frames:
        include = True
        sentences = frame.getSentences()
        # prune frames depending on no_sentences
        if filter.__contains__("min_sentences"):
            if len(sentences) < filter["min_sentences"]:
                include = False
        if include:
            r_frames.append(frame)
        else:
            no_filtered_frames += 1
    return (r_frames, no_filtered_frames)


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
