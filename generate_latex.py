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
        malt_id_file = 'malt_id_evaluation.txt'
        malt_label_file = 'malt_label_evaluation.txt'
        spacy_id_file = 'spacy_id_evaluation.txt'
        spacy_label_file = 'spacy_label_evaluation.txt'
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
        if descr_file in files and malt_id_file in files and malt_label_file in files and spacy_id_file in files and spacy_label_file in files:
            df = open(os.path.join(subdir, descr_file), 'r')
            first_line = df.readline().strip()
            df.close()

            # Different test groups
            if 'Testing good guess, All features except' in first_line:
                save_to_file(f'%%% {first_line} %%%\n', file)

                malt_id_metrics = parse_metrics_from_file(
                    os.path.join(subdir, malt_id_file))
                malt_label_metrics = parse_metrics_from_file(
                    os.path.join(subdir, malt_label_file))
                spacy_id_metrics = parse_metrics_from_file(
                    os.path.join(subdir, spacy_id_file))
                spacy_label_metrics = parse_metrics_from_file(
                    os.path.join(subdir, spacy_label_file))
                latex_table_string = build_latex_table(first_line, malt_id_metrics, malt_label_metrics,
                                                       spacy_id_metrics, spacy_label_metrics)

                save_to_file(latex_table_string, file)


def parse_metrics_from_file(filepath):
    f = open(filepath, 'r')
    for last_line in f:
        pass
    l = last_line.split()
    metrics = {'precision': l[2], 'recall': l[3], 'f1': l[4]}
    return (metrics)


def build_latex_table(title, malt_id_metrics: dict, malt_label_metrics, spacy_id_metrics, spacy_label_metrics):
    title = title.replace('_', '\_')
    return f"\\begin{{center}}\n \
\\begin{{table}}[H]\n \
 \\caption{{{title}}}\n \
 \\begin{{tabular}}{{| c || c c c | c c c |}}\n \
 \\hline\n \
 &\n \
 \\multicolumn{{3}}{{c}}{{Maltparser}} &\n \
 \\multicolumn{{3}}{{c|}}{{spaCy}} \\\\\n \
 \\hline\n \
 Stage & Precision & Recall & F1-score & Precision & Recall & F1-score \\\\ [0.5ex]\n \
 \\hline\\hline\n \
 Identification & {malt_id_metrics['precision']} & {malt_id_metrics['recall']} & {malt_id_metrics['f1']} & {spacy_id_metrics['precision']} & {spacy_id_metrics['recall']} & {spacy_id_metrics['f1']} \\\\\n \
 \\hline\n \
 Classification & {malt_label_metrics['precision']} & {malt_label_metrics['recall']} & {malt_label_metrics['f1']} & {spacy_label_metrics['precision']} & {spacy_label_metrics['recall']} & {spacy_label_metrics['f1']}\\\\\n \
 \\hline\n \
\\end{{tabular}}\n \
\\end{{table}}\n \
\\end{{center}}\n \
\n"


if __name__ == "__main__":
    main()
