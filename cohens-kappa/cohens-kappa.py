def cohens_kappa(rater1, rater2):
    y_true = np.asarray(rater1)
    y_pred = np.asarray(rater2)
    
    if len(y_true) == 0:
        return np.nan
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    label_map = {label: i for i, label in enumerate(classes)}
    
    cm = np.zeros((len(classes), len(classes)))
    for t, p in zip(y_true, y_pred):
        cm[label_map[t], label_map[p]] += 1
        
    n = np.sum(cm)
    if n == 0:
        return np.nan

    po = np.trace(cm) / n
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = np.sum((row_sums * col_sums)) / (n * n)

    if pe == 1:
        return 1.0
    
    return (po - pe) / (1 - pe)