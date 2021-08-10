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
def prune_sentences(sentences: List[Sentence], filter: dict, balance: bool = False) -> List[TreeNode]:
    role_words = []
    none_role_words = []
    roles = set()
    r_words = []

    prune_method = filter["prune"]

    if "prune" in filter.keys():

        # Prunes the sentence by the classical method
        if prune_method == 0:
            for sentence in sentences:
                unpruned_words = prune(sentence)
                r_words.extend(unpruned_words)

        # This method prunes the sentence the same way as previous but includes all words in all roles
        elif prune_method == 1:
            for sentence in sentences:
                fes = sentence.getFrameElements()
                lus = []
                for fe in fes:
                    if fe.getName() != "LU":
                        words_with_role = sentence.getNodesInFrameElement(fe)
                        for role_word in words_with_role:
                            roles.add(role_word.getRole())
                        role_words.extend(words_with_role)

                    else:
                        lus.extend(words_with_role)
                words_without_role = prune_from_predicates(
                    lus, no_role=True)
                none_role_words.extend(words_without_role)

        # This method returns all words in roles and no more
        elif prune_method == 2:
            for sentence in sentences:
                fes = sentence.getFrameElements()
                for fe in fes:
                    if fe.getName() != "LU":
                        words_with_role = sentence.getNodesInFrameElement(fe)
                        r_words.extend(words_with_role)
        else:
            for sentence in sentences:
                r_words.extend(sentence.getTreeNodesOrdered())
    else:
        for sentence in sentences:
            r_words.extend(sentence.getTreeNodesOrdered())

    # Merge the the words and balance the none words if asked
    if prune_method == 1:
        r_words.extend(role_words)
        if balance:
            avg_role_occurance = len(role_words)/len(roles)
            for i in range(0, len(none_role_words) - 1, int(len(none_role_words)/(avg_role_occurance*100))):
                r_words.append(none_role_words[i])
        else:
            r_words.extend(none_role_words)

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
                if not no_role and not w in predicates:
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

    # Count the frames
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
    # Filter sentences if they contain a role less than min_role_occurance
    if filter.__contains__("min_role_occurance"):
        min_role_occurance = filter["min_role_occurance"]
        decreased_role_occurance = True
        while decreased_role_occurance:
            decreased_role_occurance = False
            for frame in frames:
                decreased_role_occurance = filter_min_role(
                    frame, role_occurance, min_role_occurance)
    # Filter sentences if they contain a role greater than max_role_occurance
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


def filter_min_role(frame: Frame, role_occurance: dict, min_role_occurance: int):
    sentences = frame.getSentences()
    decreased_role_occurance = False
    for sentence in sentences:
        if check_min_role_sentence(sentence, role_occurance, min_role_occurance):
            fes = sentence.getFrameElements()
            frame.removeSentence(sentence)
            for fe in fes:
                if role_occurance.__contains__(fe.getName()):
                    role_occurance[fe.getName()] -= 1
                    decreased_role_occurance = True
    return decreased_role_occurance


def check_min_role_sentence(sentence: Sentence, role_occurance, min_role_occurance):
    fes = sentence.getFrameElements()
    role_names = [fe.getName() for fe in fes]
    for role_name in role_names:
        if role_occurance[role_name] < min_role_occurance:
            return True
    return False


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
