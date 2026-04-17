import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)

    G_then = G + g**2
    return (w - lr / np.sqrt(G_then + eps) * g), G_then