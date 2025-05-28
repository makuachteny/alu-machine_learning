#!/usr/bin/env python3

"""This module has a function that creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used
    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: list of the features used for embeddings
    """
    # Tokenize sentences and normalize words
    tokenized_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        words = sentence.lower().split()  # Convert to lowercase
        tokenized_sentences.append(words)

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(
            word for sentence in tokenized_sentences for word in sentence))
    else:
        vocab = sorted(vocab)  # Ensure consistent ordering

    # Create embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab:  # Only count words in the provided vocabulary
                embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab