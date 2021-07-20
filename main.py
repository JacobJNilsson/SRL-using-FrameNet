# from os import error
# from typing import List
from datetime import datetime
import os
from typing import List

from spacy.util import load_model
from data_parser import (
    parse,
    create_data,
    parse_malt,
    parse_syn_tree,
    parse_sem,
    parse_spacy,
    getFrame,
    compareFrames,
    combineFrameLists,
)
from data_struct import Frame, TreeNode
from prune import pruneFaltySentences, pruneFrames, prune_sentences
from format_data import (
    dict_data,
    split_data_train_test,
    split_data_to_identification_subsets,
    split_data_to_classification_subsets,
    sentence_data,
    create_feature_representation,
)
from training import (
    cross_val,
    evaluate_sentences,
    test_classifier,
    train_classifier,
    train_svm,
    test_labeler,
    test_ident,
    cross_val,
)
import time
import numpy as np
from helpers import save_to_file, timestamp, save_model, open_model
from message import send_email


def pipeline(
    directory: str,
    frames: List[Frame],
    parser_name: str,
    extract_features: set,
):

    # Add feature representation of each word to each word node
    create_feature_representation(frames, extract_features)

    # Prune sentences without an LU from the frames
    pruneFaltySentences(frames)

    # Split data into training and test sets
    (train_sentences, test_sentences) = split_data_train_test(frames)

    # Train models
    id_clf, label_clf = train_models(train_sentences, extract_features)

    # Save models
    save_model(id_clf, f"{parser_name}_identification_model", directory)
    save_model(label_clf, f"{parser_name}_labeling_model", directory)

    # Test models
    evaluation = test_models(id_clf, label_clf, test_sentences, extract_features)

    # Save evaluation
    save_to_file(evaluation, f"{directory}/{parser_name}_evaluation")

    return evaluation


def train_models(train_sentences, extract_features) -> tuple:

    # Prune the train data set
    pruned_train_words = prune_sentences(train_sentences)

    # Train on pruned sentences
    id_clf = train_classifier(
        pruned_train_words, extract_features, bool_result=True, prob=False
    )
    label_clf = train_classifier(
        pruned_train_words, extract_features, bool_result=False, prob=True
    )

    return (id_clf, label_clf)


def test_models(id_clf, label_clf, test_sentences, extract_features):
    pruned_test_words = prune_sentences(test_sentences)

    test_classifier(id_clf, pruned_test_words, extract_features, bool_result=True)

    chosen_words = []
    for w in pruned_test_words:
        if w.getPrediction == "1":
            chosen_words.append(w)

    test_classifier(label_clf, chosen_words, extract_features, bool_result=True)

    evaluation = evaluate_sentences(test_sentences)

    return evaluation


############ Malt ############
def run_malt(
    data_description,
    directory,
    use_directory,
    extract_features,
    model=None,
    send_mail=True,
):
    last_time = time.time()

    # Parse data
    malt_frames = parse_malt()

    send_email(
        data_description,
        f"Starting pipeline for data parsed with Maltparser",
        "jacobjnilsson@gmail.com",
        send_mail,
    )

    # Send the data to the pipeline
    result = pipeline(directory, malt_frames, "malt", extract_features)

    # Present data
    print(result)
    send_email(
        data_description,
        f"Pipeline for data parsed with Maltparser compleate. \nResult:\n{result}",
        "jacobjnilsson@gmail.com",
        send_mail,
    )
    timestamp(last_time, "Malt pipeline: ")


############ spaCy ############
def run_spacy(
    data_description,
    directory,
    use_directory,
    extract_features,
    model=None,
    send_mail=True,
):
    last_time = time.time()

    # Parse data
    # spacy_frames = open_model("spacy_parse", ".")
    spacy_frames = parse_spacy()
    save_model(spacy_frames, "spacy_parse", ".")

    send_email(
        data_description,
        f"Starting pipeline for data parsed with spaCy",
        "jacobjnilsson@gmail.com",
        send_mail,
    )

    # Send the data to the pipeline
    result = pipeline(directory, spacy_frames, "spacy", extract_features)

    # Present data
    print(result)
    send_email(
        data_description,
        f"Pipeline for data parsed with spaCy compleate. \nResult:\n{result}",
        "jacobjnilsson@gmail.com",
        send_mail,
    )
    timestamp(last_time, "spaCy pipeline: ")


def main():
    # Run variables
    start = time.time()
    now = datetime.now()
    dt_string = now.strftime("_%Y-%m-%d_%H-%M")
    directory = "run" + dt_string
    use_directory = "run_2021-07-15_19-56"
    readable_time = now.strftime("%H:%M:%S %Y-%m-%d")
    send_mail = True
    # Features of data to use
    features = {
        "word",
        "lemma",
        "pos",
        "deprel",
        "frame",
        "head_name",
        "head_lemma",
        "head_pos",
    }
    # Change this string to represent the data manipulation made
    data_description = (
        f"malt and spacy test, no list features. linearSVC. Time: {readable_time}"
    )

    print(f"Run started at: {readable_time}")
    send_email(
        data_description,
        f"New run started: {data_description} \nTime: {readable_time}\n",
        "jacobjnilsson@gmail.com",
        send_mail,
    )

    # C reate new run folder
    try:
        os.mkdir(directory)
    except:
        raise OSError("Unable to create directory")

    # Description of run
    f = open(directory + "/run_description.txt", "a")
    f.write(data_description)
    f.close()

    ######## RUNS ########
    run_malt(data_description, directory, use_directory, features, send_mail=send_mail)
    run_spacy(data_description, directory, use_directory, features, send_mail=send_mail)

    timestamp(start, "Total time: ")


if __name__ == "__main__":
    main()
