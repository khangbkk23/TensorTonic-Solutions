import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T)
    points = np.asarray(points)

    if points.ndim == 1:
        p = np.append(points, 1)
        p_new = T @ p
        return (p_new[:3] / p_new[3]).tolist()

    else:
        ones = np.ones((points.shape[0], 1))
        p = np.hstack([points, ones])      # (N,4)

        p_new = (T @ p.T).T                # (N,4)
        p_new = p_new[:, :3] / p_new[:, 3:4]

        return p_new.tolist()