from helpers import save_to_file
import os

# %%%% All except Frame %%%%
# \begin{center}
# \begin{table}[H]
#  \caption{All features except frame}
#  \begin{tabular}{| c || c c c | c c c |}
#  \hline
#  &
#  \multicolumn{3}{c}{Maltparser} &
#  \multicolumn{3}{c|}{spaCy} \\
#  \hline
#  Stage & Precision & Recall & F1-score & Precision & Recall & F1-score \\ [0.5ex]
#  \hline\hline
#  Identification & 0.78 & 0.78 & 0.78 & 0.79 & 0.79 & 0.79 \\
#  \hline
#  Classification & 0.55 & 0.50 & 0.50 & 0.60 & 0.54 & 0.54\\
#  \hline
# \end{tabular}
# \end{table}
# \end{center}


def main():
    file = 'latex_results.txt'
    if os.path.exists(file):
        os.remove(file)
    # read top file from input
    here = os.getcwd()
    folder = input("Enter root file from here: ")
    if not folder:
        rootdir = here
    else:
        rootdir = f"{here}/{folder}"
    # open file
    assert os.path.exists(rootdir), f"This is not an existing path: {rootdir}"
    for subdir, dirs, files in os.walk(rootdir):
        descr_file = 'run_description.txt'
        id_malt_file = 'malt_id_evaluation.txt'
        features = set()
        id_spacy_precision = ""
        id_spacy_recall = ""
        id_spacy_f1 = ""
        id_malt_precision = ""
        id_malt_recall = ""
        id_malt_f1 = ""
        label_spacy_precision = ""
        label_spacy_recall = ""
        label_spacy_f1 = ""
        label_malt_precision = ""
        label_malt_recall = ""
        label_malt_f1 = ""

        if descr_file in files:
            f = open(os.path.join(subdir, descr_file), 'r')
            # Different test groups
            if f.readline().strip() == 'Testing good guess, one feature at a time.':
                features = eval(f.readlines()[1].strip()[9:-1])
                first_line = f"%%%% Only {next(iter(features))} %%%%"
                save_to_file(first_line, file)
                f.close()
            f = open(os.path.join())

        if features:
            save_to_file()
        for file in files:
            pass
            # print(os.path.join(subdir, file))


if __name__ == "__main__":
    main()
