import re
import xml.etree.ElementTree as ET
from data_struct import Frame, Sentence, TreeNode, FrameElement
from tabulate import tabulate


def pruneRoleName(role: str) -> str:
    # Remove "." from the end of some frame names
    if role.endswith("."):
        role = role[:-1]
    return role


def parse_sem(datafile="swefn.xml"):
    frames = []
    tree = ET.parse(datafile)
    root = tree.getroot()
    lexicon = root.find("Lexicon")
    lexicalEntries = lexicon.findall("LexicalEntry")
    no_examples = 0
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
            no_examples += 1
            # Ititiate Sentence
            sentence = Sentence()
            i = 0
            for element in example:
                for subelement in element.iter():
                    role = subelement.get("name") or "None"
                    role = pruneRoleName(role)
                    text_string = subelement.text or ""
                    text_list = re.findall(r"\w+|[^\w\s]", text_string, re.UNICODE)
                    position = subelement.get("n") or "out of place"
                    number_of_words = len(text_list)
                    # Create Frame Element
                    fe = FrameElement(role, (i, i + number_of_words - 1), position)
                    i += number_of_words
                    sentence.addWords(text_list)
                    sentence.addFrameElement(fe)
                    # print(fe)
            if sentence.getSentence() != []:
                frame.addSentence(sentence)
        frames.append(frame)

    return frames


def main():
    frames = parse_sem()
    no_frames = len(frames)
    no_tot_sentences = 0
    frame_with_most_sentences = (None, 0)
    frame_with_least_sentences = (None, 99999)
    sentences_per_frame = {"Frame": [], "No. sentences": []}
    lu_per_frame = {"Frame": [], "No. LU": []}
    sentences_per_lu = {}
    for frame in frames:
        sentences = frame.getSentences()
        lu = frame.getLexicalUnits()
        no_sentences = len(sentences)
        no_lu = len(lu)
        sentences_per_frame["Frame"].append(frame.getName())
        sentences_per_frame["No. sentences"].append(no_sentences)
        lu_per_frame["Frame"].append(frame.getName())
        lu_per_frame["No. LU"].append(no_lu)
        no_tot_sentences += no_sentences
        if no_sentences > frame_with_most_sentences[1]:
            frame_with_most_sentences = (frame.getName(), no_sentences)
        if no_sentences < frame_with_least_sentences[1]:
            frame_with_least_sentences = (frame.getName(), no_sentences)
    lu_per_frame["Frame"].append("Total")
    lu_per_frame["No. LU"].append(sum(lu_per_frame["No. LU"]))

    average_sentences_per_frame = no_tot_sentences / no_frames
    # table = tabulate(
    #     sentences_per_frame, tablefmt="latex", headers=["Frame", "No. Sentences"]
    # )

    table = tabulate(lu_per_frame, tablefmt="latex", headers=["Frame", "No. LU"])

    # table = tabulate(
    #     [
    #         ["No. Frames total", no_frames],
    #         ["No. Sentences total", no_tot_sentences],
    #         ["Frame with most sentences:", ""],
    #         [frame_with_most_sentences[0], frame_with_most_sentences[1]],
    #         ["Frame with least sentences:", ""],
    #         [frame_with_least_sentences[0], frame_with_least_sentences[1]],
    #         ["Average no. sentences per frame", average_sentences_per_frame],
    #     ],
    #     tablefmt="latex",
    # )

    f = open("dataset_info.txt", "w")
    f.write(table)
    f.close()


if __name__ == "__main__":
    main()
