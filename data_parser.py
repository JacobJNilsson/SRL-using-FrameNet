import re
import spacy
import xml.etree.ElementTree as ET
import spacy.tokenizer as st

from typing import List
from data_struct import Frame, Sentence, TreeNode, FrameElement


# Fix bad roles
def pruneRoleName(role: str) -> str:
    # Remove "." from the end of some frame names
    if role.endswith("."):
        role = role[:-1]
    return role


def parse(datafile="swefn.xml"):
    tree = ET.parse(datafile)
    root = tree.getroot()
    lexicon = root.find("Lexicon")
    lexicalEntries = lexicon.findall("LexicalEntry")
    # print(type(lexicon))
    frames = 0
    framesWithExample = 0
    totalExamples = 0

    for lexicalEntry in lexicalEntries:
        frames += 1
        examples = 0
        lus = 0
        coreFE = 0
        sense = lexicalEntry.find("Sense")
        # children = []
        for child in sense:
            if (
                child.tag
                == "{http://spraakbanken.gu.se/eng/research/infrastructure/karp/karp}example"
            ):
                # children.append(child)
                examples += 1
            elif child.attrib["att"] == "LU":
                lus += 1
            elif child.attrib["att"] == "coreElement":
                coreFE += 1

        if examples > 0:
            framesWithExample += 1
            totalExamples += examples
        print(sense.attrib["id"])
        # for child in children:
        # print(
        #     child.tag, child.attrib
        # )  # Make better print or don't print at all
        print("Examples:    " + str(examples))
        print("LUs:         " + str(lus))
        print("Core FEs:    " + str(coreFE))
        print("")

    print("Number of frames:        " + str(frames))
    print("Of which had examples:   " + str(framesWithExample))
    print("Examples:                " + str(totalExamples))
    print(
        "Avg. number of examples per frame with examples: "
        + str(totalExamples / framesWithExample)
    )
    return root


def create_data(datafile_1="swefn-ex.xml", datafile_2="swefn.xml"):
    # extracting data from syntax
    word_file = open("word", "w", encoding="utf8")
    lemma_file = open("lemma", "w", encoding="utf8")
    pos_file = open("pos", "w", encoding="utf8")
    deprel_file = open("deprel", "w", encoding="utf8")
    dephead_file = open("dephead", "w", encoding="utf8")

    ne_ex_file = open("ne_ex", "w", encoding="utf8")
    ne_name_file = open("ne_name", "w", encoding="utf8")
    ne_type_file = open("ne_type", "w", encoding="utf8")
    ne_subtype_file = open("ne_subtype", "w", encoding="utf8")

    syn_tree = ET.parse(datafile_1)
    syn_root = syn_tree.getroot()

    for frame in syn_root:
        for example in frame:

            words = example.iter(tag="w")
            for word in words:
                word_file.write(word.text + "\n")
                lemma_file.write(word.get("lemma") + "\n")
                pos_file.write(word.get("pos") + "\n")
                deprel_file.write(word.get("deprel") + "\n")
                dephead_file.write(str(word.get("dephead", 0)) + "\n")

            frameElements = example.iter(tag="ne")
            for fe in frameElements:
                ne_ex_file.write(fe.get("ex") + "\n")
                ne_name_file.write(fe.get("name") + "\n")
                ne_type_file.write(fe.get("type") + "\n")
                ne_subtype_file.write(fe.get("subtype") + "\n")

            word_file.write("--\n")
            lemma_file.write("--\n")
            pos_file.write("--\n")
            deprel_file.write("--\n")
            dephead_file.write("--\n")
            ne_ex_file.write("--\n")
            ne_name_file.write("--\n")
            ne_type_file.write("--\n")
            ne_subtype_file.write("--\n")

    word_file.close()
    lemma_file.close()
    pos_file.close()
    deprel_file.close()
    dephead_file.close()
    ne_ex_file.close()
    ne_name_file.close()
    ne_type_file.close()
    ne_subtype_file.close()

    # extracting data from semantic examples
    # frame_tree = ET.parse(datafile_2)
    # frame_root = frame_tree.getroot()
    # lexicon = frame_root.find("Lexicon")
    # lexicalEntries = lexicon.findall("LexicalEntry")

    # for lexicalEntry in lexicalEntries:


def parse_syn_tree(datafile="swefn-ex.xml") -> List[Frame]:
    frames = []
    syn_tree = ET.parse(datafile)
    syn_root = syn_tree.getroot()

    # Each frame
    for text in syn_root:
        core_elements = text.get("core_elements").split("|")
        core_elements = [e.lstrip() for e in core_elements if e != ""]
        core_elements = [pruneRoleName(r) for r in core_elements]
        lexical_units = text.get("lexical_units_saldo").split("|")
        lexical_units = [e.lstrip() for e in lexical_units if e != ""]
        peripheral_elements = text.get("peripheral_elements").split("|")
        peripheral_elements = [e.lstrip() for e in peripheral_elements if e != ""]
        peripheral_elements = [pruneRoleName(r) for r in peripheral_elements]
        frame = Frame(
            name=text.get("frame"),
            core_elements=core_elements,
            lexical_units=lexical_units,
            peripheral_elements=peripheral_elements,
        )
        # Each example sentance
        for example in text:
            sentence = Sentence()
            subtrees = {}
            words = example.iter(tag="w")
            root = None

            # Each word in the sentance
            # Make a treenode for each word
            for word in words:
                deprel = word.get("deprel")
                placement = word.get("ref")
                dephead = word.get("dephead")
                lemma = word.get("lemma").split("|")
                lemma = [e for e in lemma if e != ""]

                if placement != None:
                    placement = int(placement)
                if dephead != None:
                    dephead = int(dephead)

                word_node = TreeNode(
                    word=word.text,
                    lemma=lemma,
                    pos=word.get("pos"),
                    deprel=deprel,
                    dephead=dephead,
                    ref=placement,
                )

                subtrees[placement] = word_node
                sentence.addWord(word_node, placement)
                if deprel == "ROOT":
                    root = word_node

            # Connect each treenode to its head treenode
            for ref in subtrees:
                word = subtrees[ref]
                dephead = word.getDephead()
                # Get the head treenode and add the subtree
                if dephead != None:
                    head = subtrees[dephead]
                    head.addSubtree(word)
                    word.addParent(head)
                    # subtrees[dephead] = head  # not sure if necessary

            # Sometimes ROOT does not exist.
            # Is this an expected situation or an error in the data?
            if root != None:
                sentence.addSynRoot(root)
                frame.addSentence(sentence)

        frames.append(frame)
    return frames


def parse_sem(datafile="swefn.xml"):
    frames = []
    tree = ET.parse(datafile)
    root = tree.getroot()
    lexicon = root.find("Lexicon")
    lexicalEntries = lexicon.findall("LexicalEntry")

    for lexicalEntry in lexicalEntries:
        sense = lexicalEntry.find("Sense")
        frameName = ""
        core_elements = []
        peripheral_elements = []
        lexical_units = []
        features = sense.findall("feat")

        for feature in features:
            # get frame name from feat with attribute BNFID
            if feature.get("att") == "BFNID":
                frameName = feature.get("val")
            elif feature.get("att") == "LU":
                lexical_units.append(feature.get("val"))
            elif feature.get("att") == "coreElement":
                core_elements.append(feature.get("val"))
            elif feature.get("att") == "peripheralElement":
                peripheral_elements.append(feature.get("val"))

        # If BNFID isn't available, use the sense id
        if frameName == "":
            frameName = sense.get("id").split("-")[-1]

        # Clean up the core elements
        core_elements = [e for e in core_elements if e != ""]
        core_elements = [pruneRoleName(r) for r in core_elements]

        # Clean up the peripheral elements
        peripheral_elements = [e for e in peripheral_elements if e != ""]
        peripheral_elements = [pruneRoleName(r) for r in peripheral_elements]

        # Clean up the lexical units
        lexical_units = [e for e in lexical_units if e != ""]

        # Initiate Frame
        frame = Frame(
            name=frameName,
            lexical_units=lexical_units,
            core_elements=core_elements,
            peripheral_elements=peripheral_elements,
        )
        examples = sense.findall(
            "{http://spraakbanken.gu.se/eng/research/infrastructure/karp/karp}example"
        )
        for example in examples:
            # Ititiate Sentence
            sentence = Sentence()
            i = 0
            for element in example:
                for subelement in element.iter():
                    role = subelement.get("name") or "None"
                    role = pruneRoleName(role)
                    text_string = subelement.text or ""
                    text_list = re.findall(r"\w+|[^\w\s]", text_string, re.UNICODE)
                    number_of_words = len(text_list)
                    # Create Frame Element
                    fe = FrameElement(role, (i, i + number_of_words - 1))
                    i += number_of_words
                    sentence.addWords(text_list)
                    sentence.addFrameElement(fe)
                    # print(fe)
            if sentence.getSentence() != []:
                frame.addSentence(sentence)
        # Prune frames without example sentences
        if frame.getSentences() != []:
            frames.append(frame)
    return frames


def custom_tokenizer(nlp):
    prefix_re = re.compile(r"""^[\[\($€¥£"']""")
    suffix_re = re.compile(r"""[\]\).,:;!?"']$""")
    infix_re = re.compile(r"""[-~=+_^*:]""")
    return st.Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
    )


def parse_spacy(datafile="swefn-ex.xml") -> List[Frame]:
    nlp = spacy.load("sv_pipeline")
    # Added custom tokenizer for recognizing math operands as seperate tokens ("1+2" -> "1", "+", "2")
    nlp.tokenizer = custom_tokenizer(nlp)
    frames = []
    syn_tree = ET.parse(datafile)
    syn_root = syn_tree.getroot()

    # Each frame
    for text in syn_root:
        core_elements = text.get("core_elements").split("|")
        core_elements = [e.lstrip() for e in core_elements if e != ""]
        core_elements = [pruneRoleName(r) for r in core_elements]
        lexical_units = text.get("lexical_units_saldo").split("|")
        lexical_units = [e.lstrip() for e in lexical_units if e != ""]
        peripheral_elements = text.get("peripheral_elements").split("|")
        peripheral_elements = [e.lstrip() for e in peripheral_elements if e != ""]
        peripheral_elements = [pruneRoleName(r) for r in peripheral_elements]
        frame = Frame(
            name=text.get("frame"),
            core_elements=core_elements,
            lexical_units=lexical_units,
            peripheral_elements=peripheral_elements,
        )
        # Each example sentance
        for example in text:
            sentence = Sentence()
            subtrees = {}
            words = example.iter(tag="w")  # Each word in the sentance
            root = None
            frame_elements = example.iter(tag="element")

            sentence_text = ""
            for word in words:
                pos = word.get("pos")
                if pos != "MAD" and pos != "MID" and pos != "PAD":
                    sentence_text += " "
                sentence_text += word.text
            sentence_text = sentence_text.lstrip()
            doc = nlp(sentence_text)
            print(f"Parsing sentence: {sentence_text}")
            # Make a treenode for each word
            for token in doc:
                deprel = token.dep_
                if deprel == "Root":
                    deprel = "ROOT"
                placement = token.i + 1
                dephead = None
                if not token.head == token:
                    dephead = token.head.i + 1
                lemma = token.lemma_

                word_node = TreeNode(
                    word=token.text,
                    lemma=lemma,
                    pos=token.tag_,
                    deprel=deprel,
                    dephead=dephead,
                    ref=placement,
                )

                subtrees[placement] = word_node
                sentence.addWord(word_node, placement)
                if deprel == "ROOT":
                    root = word_node
            # Connect each treenode to its head treenode
            for ref in subtrees:
                word = subtrees[ref]
                dephead = word.getDephead()
                # Get the head treenode and add the subtree
                if dephead != None:
                    head = subtrees[dephead]
                    head.addSubtree(word)
                    word.addParent(head)
                    # subtrees[dephead] = head  # not sure if necessary

            # Sometimes ROOT does not exist.
            # Is this an expected situation or an error in the data?
            if root != None:
                sentence.addSynRoot(root)
                frame.addSentence(sentence)

            for fe in frame_elements:
                name = fe.get("name")
                start = None
                end = None
                words = fe.iter(tag="w")
                for word in words:
                    if start == None:
                        i = int(word.get("ref")) - 1
                        start, end = i, i
                    else:
                        end = int(word.get("ref")) - 1
                if not start == None:
                    sentence.addFrameElement(FrameElement(name, (start, end)))
        frames.append(frame)

    return frames


def getFrame(name, frames):
    for frame in frames:
        if name == frame.getName():
            return frame
    return None


def compareFrames(frame_a, frame_b):
    if frame_a.getName() != frame_b.getName():
        return "Frame names does not match"
    frame_name = frame_a.getName()
    frame_a_sentences = frame_a.getSentences()
    frame_b_sentences = frame_b.getSentences()

    frame_a_sentence_string = ""
    frame_b_sentence_string = ""

    for s in frame_a_sentences:
        frame_a_sentence_string += "\n" + " ".join(s.getSentence())
    for s in frame_b_sentences:
        frame_b_sentence_string += "\n" + " ".join(s.getSentence())
    return (
        "Name: "
        + frame_name
        + "\nSentences a:"
        + frame_a_sentence_string
        + "\nSentences b:"
        + frame_b_sentence_string
        + "\nContains same sentences in order: "
        + str(frame_a_sentence_string == frame_b_sentence_string)
    )


def combineSentences(
    sentences_a: List[Sentence], sentences_b: List[Sentence]
) -> List[Sentence]:
    sentences_r = []
    for s_a in sentences_a:
        sentence = s_a.getSentence()
        root = None
        frameElements = []
        for s_b in sentences_b:
            if sentence == s_b.getSentence():
                root = s_a.getRoot() or s_b.getRoot()
                tree_nodes_ordered = (
                    s_a.getTreeNodesOrdered() + s_b.getTreeNodesOrdered()
                )
                frameElements = s_a.getFrameElements() + s_b.getFrameElements()
                s_r = Sentence(
                    sentence=sentence,
                    root=root,
                    frameElements=frameElements,
                    tree_nodes_ordered=tree_nodes_ordered,
                )
                sentences_r.append(s_r)
    return sentences_r


def combineFrames(frame_a: Frame, frame_b: Frame) -> Frame:
    name = frame_a.getName()
    if not frame_a.match(frame_b):
        return None

    lexical_units = frame_a.getLexicalUnits()
    if lexical_units != frame_b.getLexicalUnits():
        print("There is some discrepency with lexical units in: " + name)
        print(lexical_units)
        print(frame_b.getLexicalUnits())
        print()

    core_elements = frame_a.getCoreElements()
    if core_elements != frame_b.getCoreElements():
        print("There is some discrepency with core elements in: " + name)
        print(core_elements)
        print(frame_b.getCoreElements())
        print()

    peripheral_elements = frame_a.getPeripheralElements()
    if peripheral_elements != frame_b.getPeripheralElements():
        print("There is some discrepency with peripheral elements in: " + name)
        print(peripheral_elements)
        print(frame_b.getPeripheralElements())
        print()

    sentences = combineSentences(frame_a.getSentences(), frame_b.getSentences())
    return Frame(
        name=name,
        lexical_units=lexical_units,
        core_elements=core_elements,
        peripheral_elements=peripheral_elements,
        sentences=sentences,
    )


def combineFrameLists(frames_a: List[Frame], frames_b: List[Frame]) -> List[Frame]:
    frames_r = []
    i = 1
    for f_a in frames_a:
        # if match is true then f_a has found a match, used for frames without names
        match = False
        for f_b in frames_b:
            if f_a.match(f_b) and not match:
                f_r = combineFrames(f_a, f_b)
                frames_r.append(f_r)
                match = True
                # print(f"{i} frames has been combined. Frame combined: {f_a.getName()}")
                i += 1
    return frames_r
