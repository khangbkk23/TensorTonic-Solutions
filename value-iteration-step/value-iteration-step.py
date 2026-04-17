import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    best_val = []
    n =  len(values)
    for s in range(n):
        best = float('-inf')
        for a in range(len(transitions[s])):
            q = rewards[s][a]
            for s_next in range(n):
                q += gamma * np.sum(transitions[s][a][s_next] * values[s_next])

            if q > best:
                best = q
        best_val.append(best)
    return best_val