from data_parser import parse, create_data, parse_syn_tree, parse_sem
import data_struct as DS


def main():
    sem_frames = parse_sem()
    syn_frames = parse_syn_tree()

    print(sem_frames[1].match(syn_frames[0]))


if __name__ == "__main__":
    main()
