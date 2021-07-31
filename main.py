# from os import error
# from typing import List
from datetime import datetime
import os
from typing import List
from pandas.core import frame

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
from filter import (filter_faulty_sentences, prune_sentences, filter_data)
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
    frames: List[Frame],
    parser_name: str,
    extract_features: set,
    filter: dict,
    prune_test_data=True,
    log_data=False
):
    # Prune sentences without an LU from the frames
    # TODO: log no. sentences pruned
    filter_faulty_sentences(frames)

    # Filter frames
    (frames, no_filtered_frames_sentences) = filter_data(frames, filter)

    # Add feature representation of each word to each word node
    (no_data_points_features) = create_feature_representation(
        frames, extract_features)

    no_frames = f"Number of frames: {len(frames)}"
    print(no_frames)

    # Split data into training and test sets
    (train_sentences, test_sentences) = split_data_train_test(frames)
    no_sentences = f"Number of sentences: {len(train_sentences) + len(test_sentences)}"
    print(no_sentences)
    print(no_data_points_features)

    # Train models
    id_clf, label_clf, report_training = train_models(train_sentences, filter)

    print(f"{report_training}")
    if log_data:
        # Save models
        save_model(id_clf, f"{parser_name}_identification_model", directory)
        save_model(label_clf, f"{parser_name}_labeling_model", directory)

    # Test models
    (id_evaluation, label_evaluation, evaluation) = test_models(
        id_clf, label_clf, test_sentences, prune_test_data=prune_test_data)
    print(f"Models tested")
    if log_data:
        # Save evaluation
        save_to_file(
            id_evaluation, f"{directory}/{parser_name}_id_evaluation.txt")
        save_to_file(label_evaluation,
                     f"{directory}/{parser_name}_label_evaluation.txt")
        save_to_file(f"{evaluation}",
                     f"{directory}/{parser_name}_evaluation.txt")
        save_to_file(f"{no_frames}\n{no_sentences}\n{no_data_points_features}\n{no_filtered_frames_sentences}\n{report_training}",
                     f"{directory}/run_description.txt")

    return evaluation

############ Training ############


def train_models(train_sentences, filter) -> tuple:

    # Prune the train data set
    pruned_train_words = prune_sentences(train_sentences, filter)
    # pruned_train_words = prune_sentences_keep_role(train_sentences)
    no_data_points_identifier = f"Number of datapoints at training identifier: {len(pruned_train_words)}"
    print(no_data_points_identifier)
    # Train on pruned sentences
    id_clf, id_report = train_classifier(
        pruned_train_words, bool_result=True, prob=False
    )
    # Take a small break
    no_data_points_labeler = f"Number of datapoints at training labeler: {len(pruned_train_words)}"
    print(f"Taking 5 between training")
    time.sleep(5)
    #words = filter_for_train_classifier()
    label_clf, label_report = train_classifier(
        pruned_train_words,  bool_result=False, prob=True
    )

    return id_clf, label_clf, f"{id_report}\n{label_report}"

############ Testing ############


def test_models(id_clf, label_clf, test_sentences: List[Sentence], prune_test_data=True):
    if prune_test_data:
        test_words = prune_sentences(test_sentences)
    else:
        test_words = []
        for sentence in test_sentences:
            words = sentence.getTreeNodesOrdered()
            test_words.extend(words)
    print(f"Number of data points for identification: {len(test_words)}")

    id_evaluation = test_classifier(id_clf, test_words, bool_result=True)

    # Add identified words to the labeling test set
    identified_words = []
    for sentence in test_sentences:
        words = sentence.getTreeNodesOrdered()
        for w in words:
            prediction = w.getPrediction()
            if prediction == 1:
                # add the word to the list to be labeled
                identified_words.append(w)
            # Initialize all predictions of all words to "None"
            w.addPrediction("None")

    print(
        f"Identification compleate\nNumber of data points for labeling: {len(identified_words)}")

    label_evaluation = "No words labeled"
    if len(identified_words) > 0:
        label_evaluation = test_classifier(
            label_clf, identified_words, bool_result=False)

    # Evaluation of the compleate pipeline
    evaluation = evaluate_sentences(test_sentences)

    return (id_evaluation, label_evaluation, evaluation)


############ Malt ############
def run_malt(
    data_description,
    directory,
    extract_features,
    filter,
    model=None,
    prune_test_data=True,
):
    last_time = time.time()

    send_email(
        data_description,
        f"Starting pipeline for data parsed with Maltparser",
        email_address,
        send_mail,
    )

    # Parse data
    malt_frames = parse_malt()

    # Send the data to the pipeline
    result = pipeline(directory, malt_frames, "malt", extract_features, filter,
                      log_data=log_data, prune_test_data=prune_test_data)
    # Present data
    send_email(
        data_description,
        f"Pipeline for data parsed with Maltparser compleate. \nResult:\n{result}",
        email_address,
        send_mail,
    )
    timestamp(last_time, "Malt pipeline: ")


############ spaCy ############
def run_spacy(
    data_description,
    directory,
    extract_features,
    filter,
    model=None,
    prune_test_data=True,
):
    last_time = time.time()

    # Parse data
    spacy_frames: List[Frame] = open_model("spacy_parse", ".")
    sentences = []
    for frame in spacy_frames:
        sentences.extend(frame.getSentences())
    send_email(
        data_description,
        f"Starting pipeline for data parsed with spaCy",
        email_address,
        send_mail,
    )

    # Send the data to the pipeline
    result = pipeline(directory, spacy_frames, "spacy", extract_features, filter,
                      log_data=log_data, prune_test_data=prune_test_data)

    # Present data
    send_email(
        data_description,
        f"Pipeline for data parsed with spaCy compleate. \nResult:\n{result}",
        email_address,
        send_mail,
    )
    timestamp(last_time, "spaCy pipeline: ")


############ Main ############
def main():
    start = time.time()
    # Run variables
    # If the data should be pruned as a part of the evaluation
    pruning_test_data = True
    # Filter the data used in both training and testing
    filter = {"min_sentences": 0, "min_role_occurance": 12,
              "prune": 1}
    # Features of data to use
    features = {
        "frame",
        "word",
        "lemma",
        "pos",
        "deprel",
        "head_word",
        "head_lemma",
        "head_deprel"
        "head_pos",
        "child_word",
        "child_lemmas",
        "child_deprels",
        "child_pos",
    }

    if not os.path.isfile('spacy_parse.pkl'):
        try:
            frames = parse_spacy()
            save_model(frames, "spacy_parse", ".")
            send_email("Parsed spaCy", f"Finished parsing spaCy and saved to model",
                       email_address,
                       send_email)
        except Exception as err:
            send_email(data_description, f"Error when parsing spaCy\n{str(err)}",
                       email_address,
                       send_email)
            quit()

    ######## RUNS ########
    for i in [None, 0, 1]:
        filter["prune"] = i

        # Change this string to represent the data manipulation made
        now = datetime.now()
        dt_string = now.strftime("_%Y-%m-%d_%H-%M-%S")
        directory = f"runs/run {dt_string}"
        readable_time = now.strftime("%H:%M:%S %Y-%m-%d")
        data_description = (
            f"linearSVC. {features=}. {filter=}. {pruning_test_data=}. Time: {readable_time}"
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

        send_email(
            data_description,
            f"New run started: {data_description} \nTime: {readable_time}\n",
            email_address,
            send_mail,
        )

        run_malt(data_description, directory, features, filter,
                 prune_test_data=pruning_test_data)
        run_spacy(data_description, directory, features, filter,
                  prune_test_data=pruning_test_data)
    send_email("Tests compleate", ":)",
               email_address, send_mail)
    timestamp(start, "Total time: ")


# set these for outputs
send_mail = True
email_address = "jacobjnilsson@gmail.com"
log_data = True

if __name__ == "__main__":
    main()


# the ratio when training must be balanced
