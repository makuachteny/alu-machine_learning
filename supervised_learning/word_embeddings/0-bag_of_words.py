#!/usr/bin/env python3

"""This module has a function that
creates a bag of words embedding matrix"""
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import re


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix
    sentence - list of sentences to analyze
    vocab - list of vocab words for analysis
    returns embeddings, features
    embeddings shape - (s, f)
        s-no. of sentences in sentences
        f-mo. of features analyzed
    features - list of features for embeddings
    """

    tokenized_sentences = []
    # Tokenize sentences
    for sentence in sentences:
        # remove punctuation and split into words
        sentence = re.sub(r'[^\w\s]', '', sentence)
        words = sentence.lower().split()
        tokenized_sentences.append(words)

    # if vocab is none, all words in sentences are used
    if vocab is None:
        # create a set of all unique words in sentences
        vocab = set()
        # add all words to vocab
        for sentence in tokenized_sentences:
            vocab.update(sentence)

    # create embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(tokenized_sentences):
        # count words in sentence
        word_counts = Counter(sentence)
        for j, word in enumerate(vocab):
            # if word is in sentence add its count to embeddings
            if word in word_counts:
                embeddings[i, j] = word_counts[word]
    
    return embeddings, list(vocab)
