#!/usr/bin/env python3
"""
    This module implements the n-gram BLEU metric
"""
from math import exp


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations (each a list of words).
        sentence (list): List of words in the model-proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        float: n-gram BLEU score.
    """
    def get_grams(words, n):
        """Generate n-grams from a list of words."""
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    # Count n-grams in the sentence
    sentence_ngrams = get_grams(sentence, n)
    count_ngrams = {}
    for ng in sentence_ngrams:
        count_ngrams[ng] = count_ngrams.get(ng, 0) + 1

    # Count clipped n-grams
    count_clip = {}
    for ng in count_ngrams:
        max_count = 0
        for ref in references:
            ref_ngrams = get_grams(ref, n)
            ref_count = ref_ngrams.count(ng)
            max_count = max(max_count, ref_count)
        count_clip[ng] = min(count_ngrams[ng], max_count)

    # Calculate precision
    precision = sum(count_clip.values()) / sum(count_ngrams.values())

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    sentence_length = len(sentence)
    closest_ref_length = min(
        ref_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len)
    )
    if sentence_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1 - closest_ref_length / sentence_length)

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision
    return bleu_score
