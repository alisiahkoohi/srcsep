import numpy as np

def windows(x, window_size, stride, offset):
    """ Separate x into windows on last axis, discard any residual.
    """
    num_window = int((x.shape[-1] - window_size - offset) // stride + 1)
    windowed_x = np.stack([
        x[..., i * stride + offset:window_size + i * stride + offset]
        for i in range(num_window)
    ], -2)  # (C) x nb_w x w

    return windowed_x