import numpy as np


def square_glitch(window_size, signal_std):
    # Transient localized event, a "glitch" (to be removed).
    noise = np.zeros(window_size)
    noise[window_size // 2 - 5:window_size // 2] = -1
    noise[window_size // 2:window_size // 2 + 100] = 2
    noise[window_size // 2 + 100:window_size // 2 + 120] = -1
    noise[window_size // 2 + 120:window_size // 2 + 160] = 1
    noise /= (noise**2).mean()**0.5
    noise *= 0.5 * signal_std  #np.std(x_true)

    return noise


def aux(x):
    return 1 - 4 * x * (1 - x)


def get_glitch(window_size, a_left, a_right):
    glitch_left = a_left * np.exp(-a_left * np.arange(window_size // 2))
    glitch_right = a_right * np.exp(-a_right * np.arange(window_size // 2))
    glitch = np.concatenate([glitch_left[::-1], glitch_right])
    shift = np.random.randint(window_size)
    return np.roll(glitch, shift)


def exp_glitch(window_size, N_per_window=3, random_exponent=True):
    if random_exponent:
        a_left, a_right = 0.1 * aux(np.random.random(2)) + 0.01
    else:
        a_left, a_right = 0.1, 0.1
    glitch = np.zeros(window_size)
    for _ in range(N_per_window):
        glitch += get_glitch(window_size, a_left, a_right)
    return 2 * glitch
