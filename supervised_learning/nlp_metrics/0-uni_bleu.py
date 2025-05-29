#!/usr/bin/env python3
"""
    This module implements the uni-bleu metric 
"""
from math import exp


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence against list of references

    Args:
        references (list of list of str): List reference sentences (tokenized)
        sentence (list of str): Sentence to evaluate (tokenized).

    Returns:
        float: Unigram BLEU score
    """
    # Count words in the sentence
    count_unigram = {}
    for word in sentence:
        count_unigram[word] = count_unigram.get(word, 0) + 1

    # Clip count words based on the max references
    count_clip = {}
    for word in count_unigram:
        max_count = 0
        for ref in references:
            ref_count = ref.count(word)
            max_count = max(max_count, ref_count)
        count_clip[word] = min(count_unigram[word], max_count)

    # Calculate precision
    sum_count_clip = sum(count_clip.values())
    sum_count_unigram = sum(count_unigram.values())
    if sum_count_unigram == 0:
        return 0.0 
    precision = sum_count_clip / sum_count_unigram

    # Calculate brevity penalty
    sentence_length = len(sentence) # length of the candidate sentence
    ref_lengths = [len(ref) for ref in references] # length of ref
    
    # Find the closest reference length
    closest_ref_length = min(ref_lengths, key=lambda r: 
        (abs(r - sentence_length), r))

    if sentence_length > closest_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1 - closest_ref_length / sentence_length)

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
