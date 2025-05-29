#!/usr/bin/env python3
"""
    This module implements the cumulative BLEU metric
"""
from math import exp, log


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations 
            (each list of words)
        sentence (list): candidate sentence (list of words)
        n (int): size of the largest n-gram to use for evaluation

    Returns:
        float: cumulative BLEU score
    """

    # Helper: generate n-grams from a list of words
    def get_ngrams(words, n):
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

    # Helper: compute n-gram precision
    def ngram_precision(n):
        sentence_ngrams = get_ngrams(sentence, n)
        counts = {}
        for ng in sentence_ngrams:
            counts[ng] = counts.get(ng, 0) + 1

        clipped_counts = {}
        for ng in counts:
            max_ref_count = 0
            for ref in references:
                ref_ngrams = get_ngrams(ref, n)
                ref_counts = {}
                for ref_ng in ref_ngrams:
                    ref_counts[ref_ng] = ref_counts.get(ref_ng, 0) + 1
                if ng in ref_counts:
                    max_ref_count = max(max_ref_count, ref_counts[ng])
            clipped_counts[ng] = min(counts[ng], max_ref_count)

        total_clipped = sum(clipped_counts.values())
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return total_clipped / total

    # Compute precisions for each n-gram level
    precisions = []
    for i in range(1, n + 1):
        p = ngram_precision(i)
        if p == 0:
            return 0.0  # Any 0 precision results in BLEU = 0
        precisions.append(p)

    # Geometric mean of precisions
    log_sum = sum((1.0 / n) * log(p) for p in precisions)
    geo_mean = exp(log_sum)

    # Brevity Penalty
    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = ref_lens[0]
    for r in ref_lens:
        if abs(r - c) < abs(closest_ref_len - c):
            closest_ref_len = r
        elif abs(r - c) == abs(closest_ref_len - c) and r < closest_ref_len:
            closest_ref_len = r

    if c == 0:
        bp = 0.0
    elif c > closest_ref_len:
        bp = 1.0
    else:
        bp = exp(1 - closest_ref_len / c)

    return bp * geo_mean
