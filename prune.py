from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement


def pruneFaltySentences(frames: List[Frame]) -> None:
    for f in frames:
        sentences = f.getSentences().copy()
        for s in sentences:
            remove = True
            fes = s.getFrameElements()
            for fe in fes:
                if fe.getName() == "LU":
                    remove = False
            if remove:
                f.removeSentence(s)


def pruneFromPredicate(predicates: List[TreeNode]) -> List[TreeNode]:
    arguments = []
    for p in predicates:
        while p:
            children = p.getSubtrees()
            parent = p.getParent()
            if parent and not parent in arguments and not parent in predicates:
                arguments.append(parent)
            for c in children:
                role = c.getPos()
                if (
                    role != "MID"
                    and role != "PAD"
                    and role != "MAD"
                    and not c in predicates
                    and not c in arguments
                ):
                    arguments.append(c)
            p = parent

    return arguments


def pruneFromPredicatePreserveTree(n: TreeNode) -> TreeNode:
    predicate = n
    from_child = None
    while n != None:
        children = n.getSubtrees()

        # THIS BREAKS SOMEHOW ????? CHILDREN ARE NOT A COPY!!!!
        if from_child != None:
            children.remove(from_child)

        parent = n.getParent()
        for c in children:
            role = c.getPos()
            # If Interpunktion the child is removed
            if role == "MID" or role == "PAD" or role == "MAD":
                n.removeChild(c)
            # Else the child is wanted and its children are removed exept the node we walked from
            else:
                c.removeChildren()
                # Re-add the child we walked from
                # if from_child != None:
                #     n.addSubtree(from_child)
        from_child = n
        n = parent

    return predicate


def prune(sentence: Sentence) -> None:
    fes = sentence.getFrameElements()
    fe_lu = None
    for fe in fes:
        if fe.getName() == "LU":
            fe_lu = fe
            break
    assert fe_lu != None
    lu_range = fe_lu.getRange()
    lus = []
    for i in range(lu_range[0], lu_range[1] + 1):
        lus += [sentence.getNode(i)]

    # Get the syntactic node

    arguments = pruneFromPredicate(lus)
    sentence.addArguments(arguments)


# Return is not necessary since input is altered
def pruneSentencesInFrame(frame: Frame) -> None:
    sentences = frame.getSentences()
    for s in sentences:
        prune(s)


# Return is not necessary since input is altered
def pruneFrames(frames: List[Frame]) -> None:
    pruneFaltySentences(frames)
    for f in frames:
        pruneSentencesInFrame(f)