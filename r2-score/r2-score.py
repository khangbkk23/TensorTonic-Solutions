import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.all(y_true == y_true[0]):
        if np.allclose(y_true, y_pred):
            return 1.0
        else:
            return 0.0

    return 1.0 - float(np.sum((y_true-y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            