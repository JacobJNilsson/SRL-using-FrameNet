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
from data_struct import Frame, TreeNode, Sentence
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

############ Pipeline ############


def pipeline(
    directory: str,
    in_frames: List[Frame],
    parser_name: str,
    extract_features: set,
    filter: dict,
    prune_test_data=True,
    log_data=False
):

    frames = in_frames
    # Filter frames
    # for frame in in_frames:
    #     add_frame = True
    #     sentences = frame.getSentences()
    #     if len(sentences) >

    # Add feature representation of each word to each word node
    create_feature_representation(frames, extract_features)

    # Prune sentences without an LU from the frames
    pruneFaltySentences(frames)

    # Split data into training and test sets
    (train_sentences, test_sentences) = split_data_train_test(frames)
    # print(
    # f"Train sentences: {len(train_sentences)}\nTest sentences: {len(test_sentences)}")

    # Train models
    id_clf, label_clf = train_models(train_sentences)

    if log_data:
        # Save models
        save_model(id_clf, f"{parser_name}_identification_model", directory)
        save_model(label_clf, f"{parser_name}_labeling_model", directory)

    # Test models
    evaluation = test_models(
        id_clf, label_clf, test_sentences, prune_test_data=prune_test_data)

    if log_data:
        # Save evaluation
        save_to_file(evaluation, f"{directory}/{parser_name}_evaluation")

    return evaluation

############ Training ############


def train_models(train_sentences) -> tuple:

    # Prune the train data set
    pruned_train_words = prune_sentences(train_sentences)

    # Train on pruned sentences
    id_clf = train_classifier(
        pruned_train_words, bool_result=True, prob=False
    )
    label_clf = train_classifier(
        pruned_train_words,  bool_result=False, prob=True
    )

    return (id_clf, label_clf)

############ Testing ############


def test_models(id_clf, label_clf, test_sentences: List[Sentence], prune_test_data=True):
    # print(f"Test sentences: {len(test_sentences)}")
    if prune_test_data:
        test_words = prune_sentences(test_sentences)
    else:
        test_words = []
        for sentence in test_sentences:
            words = sentence.getTreeNodesOrdered()
            test_words.extend(words)
    print(f"Number of words to be identified: {len(test_words)}")

    test_classifier(id_clf, test_words, bool_result=True)

    identified_words = []
    for sentence in test_sentences:
        words = sentence.getTreeNodesOrdered()
        for w in words:
            prediction = w.getPrediction()
            if prediction == 1:  # add the word to the list to be labeled
                identified_words.append(w)
            else:  # set label of all words not identified as labels to "None"
                w.addPrediction("None")

    print(f"Number of words to be labeled: {len(identified_words)}")

    if len(identified_words) > 0:
        test_classifier(label_clf, identified_words, bool_result=False)

    evaluation = evaluate_sentences(test_sentences)

    return evaluation


############ Malt ############
def run_malt(
    data_description,
    directory,
    use_directory,
    extract_features,
    filter,
    model=None,
    send_mail=False,
    prune_test_data=True,
    log_data=False
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
    result = pipeline(directory, malt_frames, "malt", extract_features, filter,
                      log_data=log_data, prune_test_data=prune_test_data)

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
    filter,
    model=None,
    send_mail=True,
    prune_test_data=True,
    log_data=False
):
    last_time = time.time()

    # Parse data
    spacy_frames = open_model("spacy_parse", ".")
    # spacy_frames = parse_spacy()
    # save_model(spacy_frames, "spacy_parse", ".")

    send_email(
        data_description,
        f"Starting pipeline for data parsed with spaCy",
        "jacobjnilsson@gmail.com",
        send_mail,
    )

    # Send the data to the pipeline
    result = pipeline(directory, spacy_frames, "spacy", extract_features, filter,
                      log_data=log_data, prune_test_data=prune_test_data)

    # Present data
    print(result)
    send_email(
        data_description,
        f"Pipeline for data parsed with spaCy compleate. \nResult:\n{result}",
        "jacobjnilsson@gmail.com",
        send_mail,
    )
    timestamp(last_time, "spaCy pipeline: ")


############ Main ############
def main():
    # Run variables
    start = time.time()
    send_mail = True
    log_data = True
    pruning_test_data = False
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
    filter = {"min_sentences": 6}
    feats = [
        "word",
        "lemma",
        "pos",
        "deprel",
        "frame",
        "head_name",
        "head_lemma",
        "head_pos",
    ]
    for pruning_test_data in [False, True]:
        for f in feats:
            now = datetime.now()
            dt_string = now.strftime("_%Y-%m-%d_%H-%M-%S")
            directory = "run" + dt_string
            use_directory = "run_2021-07-15_19-56"
            readable_time = now.strftime("%H:%M:%S %Y-%m-%d")
            features = {f}
            # Change this string to represent the data manipulation made
            data_description = (
                f"linearSVC. {pruning_test_data=}. {features=}. {pruning_test_data=}. Time: {readable_time}"
            )

            print(f"Run started at: {readable_time}")
            send_email(
                data_description,
                f"New run started: {data_description} \nTime: {readable_time}\n",
                "jacobjnilsson@gmail.com",
                send_mail,
            )

            if log_data:
                # C reate new run folder
                try:
                    os.mkdir(directory)
                except:
                    raise OSError(f"Unable to create directory {directory}")

                # Description of run
                f = open(directory + "/run_description.txt", "a")
                f.write(data_description)
                f.close()

            ######## RUNS ########
            run_malt(data_description, directory, use_directory, features, filter,
                     send_mail=send_mail, prune_test_data=pruning_test_data, log_data=log_data)
            run_spacy(data_description, directory, use_directory, features, filter,
                      send_mail=send_mail, prune_test_data=pruning_test_data, log_data=log_data)

    send_email("all tests compleate", ":)",
               "jacobjnilsson@gmail.com", send_email)
    timestamp(start, "Total time: ")


if __name__ == "__main__":
    main()


#! something is wrong with pos for spacy
#! maybe something is not wrong, but it is odd that only pos as feature does not train a viable classifier
