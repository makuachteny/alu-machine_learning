#!/usr/bin/env python3

"""This module has a function that creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences):
    tokenized_sentences = []

    for sentence in sentences:
        # Remove punctuation except apostrophes (for handling possessives)
        sentence = re.sub(r"(?!\B'\b)[^\w\s']", '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip().lower()

        words = sentence.split()

        # Remove possessive 's (e.g., children's → children)
        words = [word[:-2] if word.endswith("'s") else word for word in words]

        tokenized_sentences.append(words)

    # Build sorted vocabulary
    vocab = sorted(set(word for sent in tokenized_sentences for word in sent))

    # Map words to indices
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Create bag-of-words matrix
    matrix = []
    for sentence in tokenized_sentences:
        row = [0] * len(vocab)
        for word in sentence:
            row[word_to_index[word]] += 1
        matrix.append(row)

    return matrix, vocab
