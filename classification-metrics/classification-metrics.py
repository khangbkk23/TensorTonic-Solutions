import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    support = np.sum(cm, axis=1)

    eps = 1e-12

    precision_c = TP / (TP + FP + eps)
    recall_c = TP / (TP + FN + eps)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

    if average == "micro":
        TP_total = TP.sum()
        FP_total = FP.sum()
        FN_total = FN.sum()

        precision = TP_total / (TP_total + FP_total + eps)
        recall = TP_total / (TP_total + FN_total + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

    elif average == "macro":
        precision = precision_c.mean()
        recall = recall_c.mean()
        f1 = f1_c.mean()

    elif average == "weighted":
        weights = support / (support.sum() + eps)
        precision = np.sum(weights * precision_c)
        recall = np.sum(weights * recall_c)
        f1 = np.sum(weights * f1_c)

    elif average == "binary":
        if pos_label not in class_to_idx:
            raise ValueError("pos_label not found")

        k = class_to_idx[pos_label]

        precision = precision_c[k]
        recall = recall_c[k]
        f1 = f1_c[k]

    else:
        raise ValueError("Invalid average mode")

    accuracy = TP.sum() / cm.sum()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }