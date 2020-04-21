import numpy as np


def text_features(text_data, dict_vectors, dict_word_index, meta_data):
    """
    Creates the textual features in the form of Word2Vec
    @misc{mikolov2013efficient,
    title={Efficient Estimation of Word Representations in Vector Space},
    author={Tomas Mikolov and Kai Chen and Greg Corrado and Jeffrey Dean},
    year={2013},
    eprint={1301.3781},
    archivePrefix={arXiv},
    primaryClass={cs.CL}}

    Inputs
        text_data: list - The text data in the form of files and then a list
                   for each sentence
        dict_vectors: dictionary - Holds the Word2Vec features for the corpora
        dict_word_index: dictionary - The corresponding index for the words
                         in the corpora
        meta_data: list - Holds meta data such as folder and number of words

    Output
        features: numpy.array - Feature array of a single file of text data
    """
    feature_length = meta_data[-1]
    feature_dimensions = dict_vectors[0].shape[0]
    features = np.zeros((feature_dimensions, feature_length))

    pointer = 0
    for text in text_data:
        words = text.split()
        if len(words) == 1:
            features[:, pointer] = dict_vectors[dict_word_index[words[0]]]
            pointer += 1
        else:
            for j in words:
                features[:, pointer] = dict_vectors[dict_word_index[j]]
                pointer += 1

    return features

