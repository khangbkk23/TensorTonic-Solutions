import numpy as np

def maxpool_forward(X, pool_size, stride):
    X = np.asarray(X)
    H, W = X.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((H_out, W_out))

    for i in range(H_out): 
        for j in range(W_out):
            max_val = -np.inf

            for a in range(pool_size): 
                for b in range(pool_size):
                    val = X[i * stride + a][j * stride + b]
                    if val > max_val:
                        max_val = val

            out[i][j] = max_val

    return out.tolist()