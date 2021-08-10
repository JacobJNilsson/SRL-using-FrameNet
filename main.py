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
from helpers import save_to_file, timestamp, save_model, open_model, chunks
from message import send_email

############ Pipeline ############


def pipeline(
    directory: str,
    frames: List[Frame],
    parser_name: str,
    extract_features: set,
    filter_: dict,
    prune_test_data=True,
    log_data=False
):
    filter = filter_.copy()
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

    # # For testing purpose
    # sentences = []
    # for frame in frames:
    #     sentences.extend(frame.getSentences())
    # (train_sentences, test_sentences) = (sentences, sentences)

    no_sentences = f"Number of sentences: {len(train_sentences) + len(test_sentences)}"
    print(no_sentences)
    print(no_data_points_features)

    filter = filter_.copy()

    # Train models
    id_clf, label_clf, report_training = train_models_2(
        train_sentences, filter)

    print(f"{report_training}")
    if log_data:
        model_path = f"{directory}/models"
        if not os.path.isdir(model_path):
            # Create model folder
            try:
                os.mkdir(f"{model_path}")
            except:
                raise OSError(f"Unable to create directory {model_path}")
        # Save models
        save_model(
            id_clf, f"{parser_name}_identification_model", f"{model_path}")
        save_model(
            label_clf, f"{parser_name}_labeling_model", f"{model_path}")

    # Test models
    (id_evaluation, label_evaluation, evaluation) = test_models_2(
        id_clf, label_clf, test_sentences, filter, prune_test_data=prune_test_data)
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
    train_words = prune_sentences(train_sentences, filter, balance=False)

    # Train on pruned sentences
    print(f"Number of datapoints at training identifier: {len(train_words)}")
    id_clf, id_report = train_classifier(
        train_words, bool_result=True, prob=False
    )

    train_words = prune_sentences(train_sentences, filter, balance=False)

    print(f"Number of datapoints at training labeler: {len(train_words)}")
    label_clf, label_report = train_classifier(
        train_words,  bool_result=False, prob=True
    )

    return id_clf, label_clf, f"{id_report}\n{label_report}"


def train_models_2(train_sentences, filter) -> tuple:

    # Prune the train data set
    train_words = prune_sentences(train_sentences, filter, balance=False)

    # Train on pruned sentences
    print(f"Number of datapoints at training identifier: {len(train_words)}")
    id_clf, id_report = train_classifier(
        train_words, bool_result=True, prob=False
    )

    # # Filter the words using the identification classifier similarly to testing
    # for chunk in chunks(train_words, 4):
    #     test_classifier(id_clf, chunk, bool_result=True)
    # train_words = []
    # for sentence in train_sentences:
    #     words = sentence.getTreeNodesOrdered()
    #     for w in words:
    #         prediction = w.getPrediction()
    #         if prediction == 1:
    #             # add the word to the list to be labeled
    #             train_words.append(w)

    filter["prune"] = 2
    train_words = prune_sentences(train_sentences, filter, balance=False)

    print(f"Number of datapoints at training labeler: {len(train_words)}")
    label_clf, label_report = train_classifier(
        train_words,  bool_result=False, prob=True
    )

    return id_clf, label_clf, f"{id_report}\n{label_report}"
############ Testing ############


def test_models(id_clf, label_clf, test_sentences: List[Sentence], filter, prune_test_data=True):
    if prune_test_data:
        filter["prune"] = 0
        test_words = prune_sentences(test_sentences, filter)
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

    if len(identified_words) > 0:
        label_evaluation = test_classifier(
            label_clf, identified_words, bool_result=False)
    else:
        label_evaluation = "No words labeled"

    # Evaluation of the compleate pipeline
    evaluation = evaluate_sentences(test_sentences)

    return (id_evaluation, label_evaluation, evaluation)


def test_models_2(id_clf, label_clf, test_sentences: List[Sentence], filter, prune_test_data=True):
    if prune_test_data:
        test_words = prune_sentences(test_sentences, filter)
    else:
        test_words = []
        for sentence in test_sentences:
            words = sentence.getTreeNodesOrdered()
            test_words.extend(words)
    print(f"Number of data points for identification: {len(test_words)}")

    id_evaluation = test_classifier(id_clf, test_words, bool_result=True)

    # Add identified words to the labeling test set

    filter["prune"] = 2
    argument_words = prune_sentences(test_sentences, filter)

    print(
        f"Identification compleate\nNumber of data points for labeling: {len(argument_words)}")

    if len(argument_words) > 0:
        label_evaluation = test_classifier(
            label_clf, argument_words, bool_result=False)
    else:
        label_evaluation = "No words labeled"

    # Evaluation of the compleate pipeline
    evaluation = "The totat evaluation is not applicable with these tests"

    return (id_evaluation, label_evaluation, evaluation)

############ Malt ############


def run_malt(
    data_description,
    directory,
    extract_features,
    filter,
    model=None,
    prune_test_data=False,
):
    last_time = time.time()

    send_email(
        directory,
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
        directory,
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
    prune_test_data=False,
):
    last_time = time.time()

    # Parse data
    spacy_frames: List[Frame] = open_model("spacy_parse", ".")
    sentences = []
    for frame in spacy_frames:
        sentences.extend(frame.getSentences())
    send_email(
        directory,
        f"Starting pipeline for data parsed with spaCy",
        email_address,
        send_mail,
    )

    # Send the data to the pipeline
    result = pipeline(directory, spacy_frames, "spacy", extract_features, filter,
                      log_data=log_data, prune_test_data=prune_test_data)

    # Present data
    send_email(
        directory,
        f"Pipeline for data parsed with spaCy compleate. \nResult:\n{result}",
        email_address,
        send_mail,
    )
    timestamp(last_time, "spaCy pipeline: ")


############ Main ############
def main():
    start = time.time()
    ##### Run variables #####
    # If the data should be pruned as a part of the evaluation
    pruning_test_data = True
    # Filter the data used in both training and testing
    filter = {"min_sentences": 0, "min_role_occurance": 6,
              "prune": 1}
    # Features of data to use
    features = {
        "frame",
        "core_elements",
        "word",
        "lemma",
        "pos",
        "deprel",
        "ref",
        "lu_words",
        "lu_lemmas",
        "lu_deprels"
        "lu_pos",
        "head_word",
        "head_lemma",
        "head_deprel"
        "head_pos",
        "child_words",
        "child_lemmas",
        "child_deprels",
        "child_pos",
    }

    if not os.path.isfile('spacy_parse.pkl'):
        try:
            frames = parse_spacy()
            save_model(frames, "spacy_parse", ".")
            send_email("Parsing spaCy", f"Finished parsing spaCy and saved to model",
                       email_address,
                       send_mail)
        except Exception as err:
            send_email("Parsing spaCy", f"Error when parsing spaCy\n{str(err)}",
                       email_address,
                       send_mail)
            quit()

    ######## RUNS ########
    # for feature in features_:
    #     features = {feature}
    # Change this string to represent the data manipulation made
    now = datetime.now()
    dt_string = now.strftime("_%Y-%m-%d_%H-%M-%S")
    directory = f"runs/run{dt_string}"
    readable_time = now.strftime("%H:%M:%S %Y-%m-%d")
    data_description = (
        f"Testing good guess, without parser features (dependency features). linearSVC. {features=}. {filter=}. {pruning_test_data=}. Time: {readable_time}\n"
    )

    if log_data:
        # Create new run folder
        try:
            os.mkdir(directory)
        except:
            raise OSError(f"Unable to create directory {directory}")

    # Description of run
    f = open(directory + "/run_description.txt", "a")
    f.write(data_description)
    f.close()

    send_email(
        directory,
        f"New run started: \n{data_description}\n",
        email_address,
        send_mail,
    )

    run_malt(data_description, directory, features, filter,
             prune_test_data=pruning_test_data)
    run_spacy(data_description, directory, features, filter,
              prune_test_data=pruning_test_data)

    send_email("Finished runs", "Tests compleate :)",
               email_address, send_mail)
    timestamp(start, "Total time: ")
    quit()


# set these for outputs
send_mail = True
email_address = "jacobjnilsson@gmail.com"
log_data = True

if __name__ == "__main__":
    main()


# the ratio when training must be balanced
