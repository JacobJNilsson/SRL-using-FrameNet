from __future__ import annotations
from typing import List
import copy

# Syntax tree with data
class TreeNode(object):
    def __init__(
        self,
        word: str,  # Word
        lemma: str,  # Word plain
        pos: str,  # Word role
        deprel: str,  # Relationship with parent
        dephead: int = None,  # Parent position
        parent=None,  # Parent node
        subtrees: list = [],
        ref: int = -1,  # The place of the word in the sentence
    ):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.deprel = deprel
        self.dephead = dephead
        self.parent = parent
        self.subtrees = subtrees
        self.ref = ref

    def __str__(self) -> str:
        return self.printSubtree("")

    # def __cmp__(self, other):
    #     return cmp(self.name, other.name)

    # def __copy__(self):
    #     print("__copy__()")
    #     return TreeNode(
    #         self.word,
    #         self.lemma,
    #         self.pos,
    #         self.deprel,
    #         self.dephead,
    #         self.parent,
    #         self.subtrees,
    #         self.ref,
    #     )

    # def __deepcopy__(self, memo):
    #     print("__deepcopy__(%s)" % str(memo))
    #     return TreeNode(
    #         copy.deepcopy(
    #             self.word,
    #             self.lemma,
    #             self.pos,
    #             self.deprel,
    #             self.dephead,
    #             self.parent,
    #             self.subtrees,
    #             self.ref,
    #             memo,
    #         )
    #     )

    def printSubtree(self, level) -> str:
        r = (
            level
            + str(self.ref)
            + " "
            + self.word
            + " "
            + self.pos
            + " "
            + str(self.dephead or "ROOT")
            + "\n"
        )
        for t in self.subtrees:
            r += t.printSubtree(level + "    ")
        return r

    def addSubtree(self, subtree):
        self.subtrees = self.subtrees + [subtree]

    def addParent(self, parent):
        self.parent = parent

    def getPos(self):
        return self.pos

    def getDephead(self):
        return self.dephead

    def getRef(self):
        return self.ref

    def getDeprel(self):
        return self.deprel

    def getLemma(self):
        return self.lemma

    def getSubtrees(self) -> List[TreeNode]:
        return self.subtrees

    def getParent(self) -> TreeNode:
        return self.parent

    def getAllSubtrees(self) -> list:
        r = [self]  # perhaps not return self
        subtrees = self.subtrees
        if subtrees != []:
            for t in subtrees:
                r += t.getAllSubtrees()
        return r

    def getAllParentsTreesExeptMe(self) -> list:
        r = []
        parent = self.getParent()
        subtrees = parent.getSubtrees().remove(self)
        r += parent.getAllParentsTreesExeptMe()
        for t in subtrees:
            r += t.t.getAllSubtrees()
        return r

    def getAllNodes(self) -> list:
        r = [self]  # perhaps not return self
        subtrees = self.subtrees
        if subtrees != []:
            for t in subtrees:
                r += t.getAllSubtrees()
        r += self.getAllParentsTreesExeptMe()

        return r

    def copy(self, parent=None, child=None, ref=None):
        me = TreeNode(
            word=self.word,
            lemma=self.lemma,
            pos=self.pos,
            deprel=self.deprel,
            dephead=self.dephead,
            ref=self.ref,
        )

        for n in self.subtrees:
            if n == ref:
                c = child
            else:
                c = n.copy(parent=me)
            me.addSubtree(c)

        n = self.parent
        if parent != None:
            p = parent
        elif n == None:
            p = n
        else:
            p = n.copy(child=me, ref=self)
        me.addParent(p)

        return me

    def removeChildren(self):
        self.subtrees = []

    def removeChild(self, child):
        self.subtrees.remove(child)

    def getRoot(self):
        if self.parent == None:
            return self
        else:
            return self.parent.getRoot()

    def getWord(self):
        return self.word


# A Frame element containing the type, its place in the sentence and "n"
class FrameElement(object):
    def __init__(self, name, range=(0, 0)):
        self.name = name
        self.range = range

    def __str__(self):
        return self.name + " " + str(self.range)

    def getName(self):
        return self.name

    def getRange(self):
        return self.range


# A sentence ordered in a list with the root to the sytax tree
class Sentence(object):
    def __init__(
        self,
        sentence: List[str] = [],
        root: TreeNode = None,
        tree_nodes_ordered: List[TreeNode] = [],
        frameElements: List[FrameElement] = [],
        arguments: List[TreeNode] = [],
    ):
        self.sentence = sentence  # The sentence as a string
        self.root = root  # The sentance as a syntax tree
        self.tree_nodes_ordered = tree_nodes_ordered  # A list of tree nodes in order
        self.frameElements = frameElements  # FrameNet data
        self.arguments = arguments

    def __str__(self) -> str:
        out_str = "Sentence: " + str(self.sentence)

        for frameElement in self.frameElements:
            out_str += "\n" + str(frameElement)

        out_str += "\n\nSyntactic tree:\n" + str(self.root)

        out_str += "\n\nArgument candidates:\n"

        for tt in self.arguments:
            out_str += str(tt) + "\n"
        return out_str

    def addWord(self, word_node: TreeNode, place: int = -1):
        if place == -1 or len(self.sentence) < place:
            self.sentence = self.sentence + [word_node.getWord()]
            self.tree_nodes_ordered = self.tree_nodes_ordered + [word_node]
        else:
            tmp_1 = self.sentence
            tmp_1.insert(place - 1, word_node.getWord())
            self.sentence = tmp_1
            tmp_2 = self.tree_nodes_ordered
            tmp_2.insert(place - 1, word_node)
            self.tree_nodes_ordered = tmp_2

    def addWords(self, words: list):
        self.sentence = self.sentence + words

    def addSynRoot(self, root: TreeNode):
        if self.root != None:
            print("There already is a root, something's fishy")
        self.root = root

    def addFrameElement(self, frameElement):
        self.frameElements = self.frameElements + [frameElement]

    def addArguments(self, arguments: List[TreeNode]):
        self.arguments = self.arguments + arguments

    def getSentence(self) -> List[str]:
        return self.sentence

    def getFrameElements(self) -> List[FrameElement]:
        return self.frameElements

    def getRoot(self) -> TreeNode:
        return self.root

    def getTreeNodesOrdered(self) -> List[TreeNode]:
        return self.tree_nodes_ordered

    def getArguments(self) -> List[TreeNode]:
        return self.arguments

    def getNode(self, index: int = 0) -> TreeNode:
        if not index < len(self.tree_nodes_ordered):
            print(f"Index '{index}' out of bounds: {self.sentence}")
        return self.tree_nodes_ordered[index]

    def getLU(self) -> FrameElement:
        for fe in self.frameElements:
            if fe.getName() == "LU":
                return fe


# A frame with data about the frame and example sentences
class Frame(object):
    def __init__(
        self,
        name: str,
        lexical_units: List[str] = [],
        core_elements: List[str] = [],
        peripheral_elements: List[str] = [],
        sentences: List[Sentence] = [],
    ):
        self.name = name  # The name of the frame
        self.lexical_units = lexical_units  # Its lexical units
        self.core_elements = core_elements  # Its core elements
        self.peripheral_elements = peripheral_elements  # Its peripheral elements
        self.sentences = sentences  # The sentance examples of the frame

    def __str__(self) -> str:
        r_str = "Name: " + self.name

        r_str += "\n\nLexical units:"
        i = 0
        for lexical_unit in self.lexical_units:
            r_str += "\n(" + str(i) + ") " + str(lexical_unit)
            i += 1

        r_str += "\n\nCore elements:"
        i = 0
        for core_element in self.core_elements:
            r_str += "\n(" + str(i) + ") " + str(core_element)
            i += 1

        r_str += "\n\nPeripheral elements:"
        i = 0
        for peripheral_element in self.peripheral_elements:
            r_str += "\n(" + str(i) + ") " + str(peripheral_element)
            i += 1

        r_str += "\n\nSentences:"
        j = 0
        for sentence in self.sentences:
            r_str += "\n(" + str(j) + ")\n" + str(sentence)
            j += 1

        return r_str

    def addLexicalUnit(self, lexical_unit: str) -> None:
        self.lexical_units = self.lexical_units + [lexical_unit]

    def addCoreElement(self, core_element: str) -> None:
        self.core_elements = self.core_elements + [core_element]

    def addPeripheralElement(self, peripheral_element: str) -> None:
        self.peripheral_elements = self.peripheral_elements + [peripheral_element]

    def addSentence(self, sentence: Sentence) -> None:
        self.sentences = self.sentences + [sentence]

    def addName(self, name: str) -> None:
        self.name = name

    def removeSentence(self, sentence: Sentence) -> None:
        self.sentences.remove(sentence)

    def getName(self) -> str:
        return self.name

    def getLexicalUnits(self):
        return self.lexical_units

    def getCoreElements(self) -> List[str]:
        return self.core_elements

    def getPeripheralElements(self) -> List[str]:
        return self.peripheral_elements

    def getSentences(self) -> List[Sentence]:
        return self.sentences

    def match(self, frame) -> bool:
        return (
            (self.name == frame.getName() or self.name == "" or frame.getName() == "")
            and self.lexical_units == frame.getLexicalUnits()
            and self.core_elements == frame.getCoreElements()
            and self.peripheral_elements == frame.getPeripheralElements()
        )


# A bad syntax tree printer
def inorder(tree: TreeNode):
    if tree is not []:
        print(
            '"'
            + tree.word
            + '"'
            + " Position: "
            + str(tree.ref)
            + " Role: "
            + tree.pos
            + " Parent: "
            + str(tree.dephead)
            + " Relation: "
            + tree.deprel
        )
        for subtree in tree.subtrees:
            inorder(subtree)


# A small example of the syntax tree and the printer
if __name__ == "__main__":
    a = TreeNode("Nästa", "|nästa|", "JJ", "test")
    b = TreeNode("år", "|år|", "NN", "test", subtrees=[a])
    d = TreeNode("svenskar", "|svensk|", "NN", "test")
    e = TreeNode("betala", "|betala|", "VB", "test")
    c = TreeNode("får", "|få|", "VB", "test", subtrees=[b, d, e])
    inorder(c)
