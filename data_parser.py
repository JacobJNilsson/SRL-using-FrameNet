import xml.etree.ElementTree as ET
import os
import data_struct as DS


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
    # script_dir = os.path.dirname(__file__)
    # rel_path = "2091/data.txt"
    # abs_file_path = os.path.join(script_dir, rel_path)

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


def parse_syn_tree(datafile="swefn-ex.xml"):
    frames = []
    syn_tree = ET.parse(datafile)
    syn_root = syn_tree.getroot()

    # Each frame
    for text in syn_root:
        frame = DS.Frame(text.get("frame"), text.get("core_elements").split("|"))
        # Each example sentance
        for example in text:
            sentence = DS.Sentence()
            subtrees = {}
            words = example.iter(tag="w")
            root = None

            # Each word in the sentance
            # Make a treenode for each word
            for word in words:
                deprel = word.get("deprel")
                placement = word.get("ref")
                dephead = word.get("dephead")

                if placement != None:
                    placement = int(placement)
                if dephead != None:
                    dephead = int(dephead)

                word_tree = DS.TreeNode(
                    word=word.text,
                    lemma=word.get("lemma"),
                    pos=word.get("pos"),
                    deprel=deprel,
                    dephead=dephead,
                    ref=placement,
                )

                subtrees[placement] = word_tree
                sentence.addWord(word.text, placement)
                if deprel == "ROOT":
                    root = word_tree

            # Connect each treenode to its head treenode
            for ref in subtrees:
                word = subtrees[ref]
                dephead = word.getDephead()
                # Get the head treenode and add the subtree
                if dephead != None:
                    head = subtrees[dephead]
                    head.addSubtree(word)
                    subtrees[dephead] = head  # not sure if necessary

            # Sometimes ROOT does not exist.
            # Is this an expected situation or an error in the data?
            if root != None:
                sentence.addSynTree(root)
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
        features = sense.findall("feat")
        for feature in features:
            # get frame name from feat with attribute BNFID
            if feature.get("att") == "BFNID":
                frameName = feature.get("val")
            elif feature.get("att") == "coreElement":
                core_elements.append(feature.get("val"))

        frame = DS.Frame(frameName, core_elements)  # Initiate Frame
        examples = sense.findall(
            "{http://spraakbanken.gu.se/eng/research/infrastructure/karp/karp}example"
        )
        for example in examples:
            sentence = DS.Sentence()  # Ititiate Sentence
            i = 0
            for element in example:
                for subelement in element.iter():
                    role = subelement.get("name") or "None"
                    text = subelement.text or ""
                    text = text.split()
                    position = subelement.get("n") or "out of place"
                    number_of_words = len(text)
                    # Create Frame Element
                    fe = DS.FrameElement(role, (i, i + number_of_words - 1), position)
                    i += number_of_words
                    sentence.addWords(text)
                    sentence.addFrameElement(fe)
                    # print(fe)
            frame.addSentence(sentence)
        frames.append(frame)
    return frames
