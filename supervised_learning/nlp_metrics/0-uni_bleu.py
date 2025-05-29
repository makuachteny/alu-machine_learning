"""
    this module implements the uni-bleu metric 
"""


def uni_bleu(references, sentence):
    """_summary_
    calculates the unigram BLEU score for a sentence against a list of 
    references 
    Args:
        references (list): list of reference sentences
        sentence (str): sentence to evaluate
    Returns:
        float: unigram BLEU score
    """
    

    # count words in the sentence
    word_counts = {}
    for word in sentence.split():
        word_counts[word] = word_counts.get(word, 0) + 1
        
    # Clip words based on references
    clipped_counts = {}
    for ref in references:
        ref_counts = {}
        for word in ref.split():
            ref_counts[word] = ref_counts.get(word, 0) + 1
            
        for word, count in word_counts.items():
            if word in ref_counts:
                clipped_counts[word] = min(count, ref_counts[word])
            else:
                clipped_counts[word] = 0
    # Calculate precision
    total_clipped = sum(clipped_counts.values())
    total_words = sum(word_counts.values())
    if total_words == 0:
        return 0.0
    precision = total_clipped / total_words
    
    # Calculate brevity penalty
    ref_lengths = [len(ref.split()) for ref in references]
    sentence_length = len(sentence.split())
    closest_ref_length = min(ref_lengths, key=lambda x: abs(x - sentence_length))
    if sentence_length > closest_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = 0.0
    # Calculate BLEU score
    bleu_score = precision * brevity_penalty
    return bleu_score
