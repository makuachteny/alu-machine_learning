#!/usr/bin/env python3

"""This module has a function that creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    sentences (list): List of sentences to analyze.
    vocab (list): List of vocabulary words to use for the analysis.
    If None, all words within sentences are used.

    Returns:
    tuple: embeddings (numpy.ndarray), features (list)
        embeddings shape - (s, f)
            s - number of sentences in sentences
            f - number of features analyzed
        features - list of features used for embeddings
    """
    # Tokenize sentences and remove punctuation
    tokenized_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^\w\s]', '', sentence)  # remove punctuation
        words = sentence.lower().split()  # lowercase
        tokenized_sentences.append(words)

    # If vocab is None, use all unique words in sentences
    if vocab is None:
        vocab = sorted(
            set(word for sentence in tokenized_sentences for word in sentence))
    else:
        # Ensure vocab is a list of unique words
        vocab = sorted(set(vocab))

    # Create embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab  # Return vocab as features
