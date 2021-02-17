# A frame with data about the frame and example sentences
class Frame(object):
    def __init__(
        self,
        name: str,
        core_elements: list = [],
        sentences: list = [],
    ):
        self.name = name  # The name of the frame
        self.core_elements = core_elements  # Its core elements
        self.sentences = sentences  # The sentance examples of the frame

    def __str__(self) -> str:
        r_str = "Name: " + self.name

        r_str += "\nCore elements:"
        i = 0
        for core_element in self.core_elements:
            r_str += "\n(" + str(i) + ") " + str(core_element)

        r_str += "\nSentences:"
        j = 0
        for sentence in self.sentences:
            r_str += "\n(" + str(j) + ") " + str(sentence)

        return r_str

    def addSentence(self, sentence: str):
        self.sentences = self.sentences + [sentence]

    def addCoreElement(self, core_element: str):
        self.core_elements = self.core_elements + [core_element]

    def getSentences(self):
        return self.sentences

    def getName(self):
        return self.name

    def match(self, frame):
        return self.name == frame.getName()


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
    def __init__(self, sentence=[], root=None, frameElements=[]):
        self.sentence = sentence  # The sentence as a string
        self.root = root  # The sentance as a syntax tree
        self.frameElements = frameElements

    def __str__(self):
        fe_str = ""
        for frameElement in self.frameElements:
            fe_str += "\n" + str(frameElement)
        return "sentence: " + str(self.sentence) + fe_str

    def addWord(self, word: str, place: int = -1):
        if place == -1 or len(self.sentence) < place:
            self.sentence = self.sentence + [word]
        else:
            tmp = self.sentence
            tmp.insert(place - 1, word)
            self.sentence = tmp

    def addWords(self, words: list):
        self.sentence = self.sentence + words

    def addSynTree(self, root):
        if self.root != None:
            print("There already is a root, something's fishy")
        self.root = root

    def addFrameElement(self, frameElement):
        self.frameElements = self.frameElements + [frameElement]

    def getSentence(self):
        return self.sentence

    def getFrameElements(self):
        return self.frameElements


class FrameElement(object):
    def __init__(self, name, range=(0, 0), pos=None):
        self.name = name
        self.range = range
        self.pos = pos

    def __str__(self):
        return "[" + self.pos + "] " + self.name + " " + str(self.range)


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
