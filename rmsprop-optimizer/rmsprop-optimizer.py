import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)

    s_then = beta * s + (1-beta)*np.power(g,2)
    w_then = w - lr * g / (np.sqrt(s_then) + eps)

    return w_then, s_then