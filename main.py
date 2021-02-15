from data_parser import parse, create_data, parse_syn_tree
import data_struct as DS


def main():
    # parse()
    # create_data()
    frames = parse_syn_tree()
    for frame in frames:
        if frame.name == "Others_situation_as_stimulus":
            for sentence in frame.getSentences():
                DS.inorder(sentence.root)
                print("")


if __name__ == "__main__":
    main()
