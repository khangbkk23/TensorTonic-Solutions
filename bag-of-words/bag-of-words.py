import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vocab_idx = {word: i for i, word in enumerate(vocab)} # map: word -> idx
    vector = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in vocab_idx:
            vector[vocab_idx[token]] += 1

    return vector

    