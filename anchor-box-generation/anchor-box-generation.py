import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size

    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios)

    w = (scales[:, None] * np.sqrt(aspect_ratios)[None, :]).reshape(-1)
    h = (scales[:, None] / np.sqrt(aspect_ratios)[None, :]).reshape(-1)

    shifts = (np.arange(feature_size) + 0.5) * stride
    cx, cy = np.meshgrid(shifts, shifts)

    cx = cx.reshape(-1, 1)
    cy = cy.reshape(-1, 1)

    w = w.reshape(1, -1)
    h = h.reshape(1, -1)

    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2

    anchors = np.stack([x_min, y_min, x_max, y_max], axis=-1)
    # shape: (num_cells, num_anchors_per_cell, 4)

    anchors = anchors.reshape(-1, 4)
    return anchors.tolist()