import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    values, counts = np.unique(y_train, return_counts=True)
    major = values[np.argmax(counts)]
    return np.full(len(X_test), major)