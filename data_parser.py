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
