import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape

    H_out = H - KH + 1
    W_out = W_in - KW + 1 
    y = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    val = 0.0
                    for c_in in range(C_in):
                       for u in range(KH):
                            for v in range(KW):
                                val += (x[n, c_in, u + i, v + j] * W[c_out, c_in, u, v])
                    val += b[c_out]
                    y[n,c_out, i, j] = val
    return y