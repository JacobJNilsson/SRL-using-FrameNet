import xml.etree.ElementTree as ET


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


def create_data(datafile="swefn-ex.xml"):
    word_file = open("word", "w", encoding="utf8")
    lemma_file = open("lemma", "w", encoding="utf8")
    pos_file = open("pos", "w", encoding="utf8")
    deprel_file = open("deprel", "w", encoding="utf8")
    dephead_file = open("dephead", "w", encoding="utf8")

    tree = ET.parse(datafile)
    root = tree.getroot()

    for frame in root:
        for example in frame:
            words = example.iter(tag="w")
            for word in words:
                word_file.write(word.text + "\n")
                lemma_file.write(word.get("lemma") + "\n")
                pos_file.write(word.get("pos") + "\n")
                deprel_file.write(word.get("deprel") + "\n")
                dephead_file.write(str(word.get("dephead", 0)) + "\n")

            word_file.write("\n")
            lemma_file.write("\n")
            pos_file.write("\n")
            deprel_file.write("\n")
            dephead_file.write("\n")

    word_file.close()
    lemma_file.close()
    pos_file.close()
    deprel_file.close()
    dephead_file.close()
