import psutil
from config_files import config, config_process
import os
from shutil import copyfile, rmtree
import sys
from utils import utilities, file_analysis as fa
import numpy as np
import h5py
from text import text_feature_extractor
import pickle
import gc
import math
from gensim.models import Word2Vec


def max_min_values(per_file_text, current_directory):
    """
    Calculates the maximum and minimum length of audio in the dataset

    Inputs
        current_directory: str - The location of the current workspace
        per_file_text: list - Contains a list  of lists holding every
                       sentence from every text file in the dataset

    Outputs
        max_value: int - The longest file in the database
        min_value: int - The shortest file in the database
        output_data: list - Contains the folder and number of words for each
                     file
    """
    print('Processing Text Files\n')
    max_value = number_sentences_max = 0
    number_sentences_min = 5
    min_value = 1e20
    output_data = np.zeros((len(per_file_text), 2), dtype=int)
    folder_number = 300
    length_of_sentences = {}
    for iterator, text_data in enumerate(per_file_text):
        print(f"Interview file is {folder_number}")
        number_words = 0
        for pointer, words in enumerate(text_data):
            length = len(words.split())
            if length not in length_of_sentences.keys():
                length_of_sentences[length] = 1
            else:
                length_of_sentences[length] += 1
            number_words += length
            if length > number_sentences_max:
                number_sentences_max = length
            if length < number_sentences_min:
                number_sentences_min = length
        if number_words > max_value:
            max_value = number_words
        if number_words < min_value:
            min_value = number_words

        output_data[iterator, :] = [folder_number, number_words]
        folder_number += 1
        if folder_number in config_process.excluded_sessions:
            folder_number += 1

    path = os.path.join(current_directory, 'meta_data')
    np.save(path, output_data)
    utilities.save_pickle(length_of_sentences, 'length_of_sentences',
                          current_directory)

    return max_value, min_value, output_data


def build_word_model(per_file_text, dict_words):
    """
    Creates the text features from strings of text data using Word2Vec

    Inputs
        per_file_text: list - Contains a list of text data for every file
        dict_words: dictionary - Holds every unique word in the corpus

    Output
        dict_vectors: dictionary - Holds the Word2Vec features for every word
                      in the corpus
    """

    corpus = [words for i in per_file_text for words in i]
    corpus = [words.split() for words in corpus]

    model = Word2Vec(corpus, size=200, window=5, min_count=1, workers=4, sg=1)
    word_vectors = model.wv
    del model

    # nlp = spacy.load('en_core_web_md')
    # # Create an empty dictionary
    # dict_vectors = dict.fromkeys(range(len(dict_words)))
    # for i in range(len(dict_words)):
    #     try:
    #         test_vec = nlp(dict_words[i]).vector
    #         dict_vectors[i] = test_vec
    #     except:
    #         print('an error occured at:', i)
    #         sys.exit(f"Specific word {dict_words[i]} was not found in nlp")

    dict_vectors = dict.fromkeys(range(len(dict_words)))
    for i in range(len(dict_words)):
        try:
            test_vec = word_vectors[dict_words[i]]
            dict_vectors[i] = test_vec
        except:
            print('an error occurred at:', i)
            sys.exit(f"Specific word {dict_words[i]} was not found in nlp")

    return dict_vectors


def create_database(main_logger, labels, current_directory, per_file_text,
                    dict_words, dict_words_index, meta_data):
    """
    Creates a database of extracted features from the raw input data such as
    text or audio. The database contains metadata such as folder, class,
    score, and the index of each respective file.

    Inputs
        main_logger: logger - Records important information
        labels: list - Holds meta data for database including folder, class,
                score, and index
        current_directory: str - The location of the save directory
        per_file_text: list - Contains every file and every sentence of
                       text data in a list per file
        dict_words: dictionary - The corpora of words and their keys
        dict_words_index: dictionary - The index of words in the dataset

    Output
        num_samples_feature: list - Records the length of every file to be
                             recorded in another function along with summary
                             data
    """
    # Everything in this h5 dataset is in numerical order. All the
    # folders, classes and features will be written in ascending order.
    # The train, dev and test labels will provide the shuffling
    datatype = h5py.special_dtype(vlen=np.float32)
    h5file = h5py.File(os.path.join(current_directory,
                                    f"complete_database.h5"), 'w')
    num_files = len(labels[0])
    h5file.create_dataset(name='folder',
                          data=[0] * num_files,
                          dtype=np.int16)
    h5file.create_dataset(name='class',
                          data=[0] * num_files,
                          dtype=np.int8)
    h5file.create_dataset(name='score',
                          data=[0] * num_files,
                          dtype=np.int8)
    h5file.create_dataset(name='index',
                          data=[0] * num_files,
                          dtype=np.int16)
    h5file.create_dataset(name='features',
                          compression="gzip",
                          shape=(num_files, 1),
                          maxshape=(num_files, None),
                          dtype=datatype)

    print('\nCreating Word Vectors\n')

    # Feature extractor
    # Loop through all the files and get the sampling rate, number of
    # samples and the time in minutes
    dict_vectors = build_word_model(per_file_text, dict_words)
    utilities.save_pickle(dict_vectors, 'dict_vectors', current_directory)
    num_samples_feature = []
    for pointer, text_data in enumerate(per_file_text):
        current_meta_data = meta_data[pointer]
        text_features = text_feature_extractor.text_features(text_data,
                                                             dict_vectors,
                                                             dict_words_index,
                                                             current_meta_data)

        print(f"Folder Name: {current_meta_data[0]}, dimensions are:"
              f" {text_features.shape[0]}, {text_features.shape[1]}")
        main_logger.info(f"Successfully created text features for the file:"
                         f" {current_meta_data[0]}, it's dimensions are:"
                         f" {text_features.shape[0]}, {text_features.shape[1]}")
        height, width = text_features.shape
        new_length = width * height
        text_features = np.reshape(text_features, new_length)
        num_samples_feature.append(width)

        clss = labels[1][pointer]
        if math.isnan(labels[2][pointer]):
            scre = -1
        else:
            scre = labels[2][pointer]

        h5file['features'][pointer] = text_features
        h5file['folder'][pointer] = current_meta_data[0]
        if current_meta_data[0] in config_process.wrong_labels:
            h5file['class'][pointer] = config_process.wrong_labels[current_meta_data[0]]
        else:
            h5file['class'][pointer] = clss
        h5file['score'][pointer] = scre
        h5file['index'][pointer] = pointer

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss
        memory = memory / 1e9
        if memory > 13:
            gc.collect()

    h5file.close()
    return num_samples_feature


def process_organise_data(main_logger,
                          current_directory):
    """
    Top level function to process the dataset by filtering out the virtual
    agent's speech from the text files and determine the meta information
    such as the folder, longest and shortest files, and per file number of
    words. It then computes the features specified for the experiment and
    saves the results in a database ready for model training.

    Inputs
        main_logger: logger - Records the important information
        current_directory: str - The location of the features folder
    """
    dataset_path = config.DATASET
    workspace_files_dir = config.WORKSPACE_FILES_DIR

    main_logger.info(f"The feature dir: {current_directory}")
    main_logger.info(f"The dataset dir: {dataset_path}")

    folder_list, audio_paths, transcript_paths = fa.get_meta_data(dataset_path)

    per_file_text, dict_words, dict_words_index = \
        utilities.transcript_text_processing(transcript_paths,
                                             current_directory)

    main_logger.info(f"The on_off_times are: {dict_words}")
    # FOR DEBUGGING USE FILES 6:9 IN ORDER TO GET ONE CLASS "1"s
    max_value, min_value, meta_data = max_min_values(per_file_text,
                                                     current_directory)
    print('max_value is: ', max_value)
    print('min_value is: ', min_value)
    main_logger.info(f"The max length (number of words) of the transcript is: "
                     f"{max_value}, the minimum is: {min_value}")

    labels = utilities.get_labels_from_dataframe(config.COMP_DATASET_PATH)

    # For debugging purposes
    # l1 = labels[0][0:2]
    # l2 = labels[1][0:2]
    # l3 = labels[2][0:2]
    # l1 = labels[0][0:35]
    # l2 = labels[1][0:35]
    # l3 = labels[2][0:35]
    # labels = [l1, l2, l3]

    num_samples_feature = create_database(main_logger,
                                          labels,
                                          current_directory,
                                          per_file_text,
                                          dict_words,
                                          dict_words_index,
                                          meta_data)

    summary_labels = ['MaxWords',
                      'MinWords',
                      'NumberFiles',
                      'WordsPerFile']

    summary_values = [max_value,
                      min_value,
                      len(labels[0]),
                      num_samples_feature]

    save_path = os.path.join(current_directory, 'summary.pickle')
    with open(save_path, 'wb') as f:
        summary = [summary_labels, summary_values]
        pickle.dump(summary, f)

    copyfile(workspace_files_dir + '/' + 'config'+'.py',
             current_directory + '/' + 'config'+'.py')


def startup():
    """
    Starter function to create the working directory and then to process the
    dataset.
    """
    features_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    workspace = config.WORKSPACE_MAIN_DIR
    folder_name = config.FOLDER_NAME
    current_directory = os.path.join(workspace, folder_name)

    # THIS WILL DELETE EVERYTHING IN THE CURRENT WORKSPACE #
    if os.path.exists(current_directory):
        option = input('A directory at this location exists, do you want '
                       'to delete? ')
        if option == ('y' or 'Y' or 'yes' or 'Yes'):
            rmtree(current_directory, ignore_errors=False, onerror=None)
        else:
            print('Please choose a different path, program will now '
                  'terminate')
            sys.exit()

    os.mkdir(current_directory)
    utilities.create_directories(current_directory,
                                 config.FEATURE_FOLDERS)
    main_logger = utilities.setup_logger(current_directory)
    main_logger.info(f"The workspace: {workspace}")
    main_logger.info(f"The experiment dir is: {features_exp}")
    process_organise_data(main_logger,
                          current_directory)

    handlers = main_logger.handlers[:]
    for handler in handlers:
        handler.close()
        main_logger.removeHandler(handler)
