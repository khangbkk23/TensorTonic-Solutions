import numpy as np

def entropy_node(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    return float(max(entropy, 0.0))