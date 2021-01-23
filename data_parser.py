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
        j = 0
        sense = lexicalEntry.find("Sense")
        children = []
        for child in sense:
            if (
                child.tag
                == "{http://spraakbanken.gu.se/eng/research/infrastructure/karp/karp}example"
            ):
                children.append(child)
                j += 1
        if j > 0:
            framesWithExample += 1
            totalExamples += j
            print(lexicalEntry.tag, lexicalEntry.attrib)
            # for child in children:
            # print(
            #     child.tag, child.attrib
            # )  # Make better print or don't print at all
            print(j)

    print("Number of frames: " + str(frames))
    print("Of which had examples: " + str(framesWithExample))
    print(
        "Avg. example per frame with example: " + str(totalExamples / framesWithExample)
    )
    return root
