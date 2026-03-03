import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    # trapezoidal rule
    delta_fpr = fpr[1:] - fpr[:-1]
    avg_tpr = (tpr[1:] + tpr[:-1]) / 2.0

    return float(np.sum(avg_tpr * delta_fpr))