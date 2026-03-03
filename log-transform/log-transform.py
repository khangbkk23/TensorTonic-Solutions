import numpy as np
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    data = np.asarray(values)
    try:
        log_transformed = np.log(data)
        print(f"Log transformed (log): {log_transformed}")
    except Exception as e:
        print(f"numpy.log error: {e}")

    log1p_transformed = np.log1p(data)
    print(f"Log transformed (log1p): {log1p_transformed}")
    return log1p_transformed