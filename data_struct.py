# Syntax tree with data
class TreeNode(object):
    def __init__(
        self,
        word: str,
        lemma: str,
        pos: str,
        deprel: str,
        dephead: int = None,
        subtrees: list = [],
        ref: int = -1,  # The place of the word in the sentence
    ):
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.deprel = deprel
        self.dephead = dephead
        self.subtrees = subtrees
        self.ref = ref

    def addSubtree(self, subtree):
        self.subtrees = self.subtrees + [subtree]

    def getDephead(self):
        return self.dephead

    def getRef(self):
        return self.ref


# A sentence ordered in a list with the root to the sytax tree
class Sentence(object):
    def __init__(self, sentence=[], root=None):
        self.sentence = sentence  # The sentence as a string
        self.root = root  # The sentance as a syntax tree

    def addWord(self, word: str, place: int = -1):
        if place == -1 or len(self.sentence) < place:
            self.sentence.append(word)
        else:
            self.sentence.insert(place - 1, word)

    def addSynTree(self, root):
        self.root = root

    def getSentence(self):
        return self.sentence


# A frame with data about the frame and example sentences
class Frame(object):
    def __init__(self, name, core_elements=[], sentences=[]):
        self.name = name  # The name of the frame
        self.core_elements = core_elements  # Its core elements
        self.sentences = sentences  # The sentance examples of the frame

    def addSentence(self, sentence):
        self.sentences = self.sentences + [sentence]

    def addCoreElement(self, core_element):
        self.core_elements = self.core_elements + [core_element]

    def getSentences(self):
        return self.sentences


# A bad syntax tree printer
def inorder(tree):
    if tree is not []:
        print(tree.word)
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
