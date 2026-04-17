import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = y_true.shape[0]
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    probs = []
    for i in range(N):
        true_class = y_true[i]
        prob = y_pred[i][true_class]
        probs.append(prob)
    return -np.mean(np.log(probs))