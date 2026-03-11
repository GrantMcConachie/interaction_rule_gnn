"""
Utility functions specifically for making fish graphs
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def generate_x_dot(loc, smooth=True, frame_rate=120, smooth_window=21, smooth_polyorder=3):
    """
    generates x dot data from location data
    """
    # init
    x = []
    x_dot = []
    x_dot_dot = []

    for i, fish in enumerate(loc):
        # averaging over keypoints
        pos = np.mean(fish, axis=1)

        # linearaly interpolating over nans
        pos = np.array(
            pd.DataFrame(pos).interpolate(axis=1, limit_direction='both')
        )

        # px to mm, center, and normalize
        pos *= (300.0 / 750.0)
        pos -= 150.0
        pos /= 150.0

        # calculating velocity and acceleration
        if smooth:
            pos = savgol_filter(pos, smooth_window, smooth_polyorder, deriv=0, axis=1)
            vel = savgol_filter(pos, smooth_window, smooth_polyorder, deriv=1, delta=1/frame_rate, axis=1)
            acc = savgol_filter(pos, smooth_window, smooth_polyorder, deriv=2, delta=1/frame_rate, axis=1)
        else:
            vel = np.gradient(pos, axis=1) * frame_rate
            acc = np.gradient(vel, axis=1) * frame_rate

        x.append(pos)
        x_dot.append(vel)
        x_dot_dot.append(acc)

    return {
        'x': np.array(x),
        'x_dot': np.array(x_dot),
        'x_dot_dot': np.array(x_dot_dot),
        'dt': 1/frame_rate
    }


def animate_fish(state_vectors, save_path, smooth_status, start=0, stop=3000, step=4, interval=33):
    """
    Animates fish positions over time and saves as a GIF.

    :param state_vectors: dict returned by generate_x_dot, with 'x' of shape (n_fish, 2, T)
    :param save_path: path to save the GIF
    :param start: first frame index
    :param stop: last frame index (None = end of data)
    :param step: stride between animated frames (higher = smaller file)
    :param interval: milliseconds between frames in the GIF
    """
    p_t = state_vectors['x']  # (n_fish, 2, T)
    n = p_t.shape[0]
    T = p_t.shape[2]
    stop = T if stop is None else min(stop, T)
    frames = range(start, stop, step)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='none', lw=1.5))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Fish trajectories (smoothed={str(smooth_status)})')

    scat = ax.scatter(p_t[:, 0, start], p_t[:, 1, start], s=30, c=np.arange(n), cmap='tab10')

    buf = np.empty((n, 2), dtype=p_t.dtype)

    def update(t):
        buf[:, 0] = p_t[:, 0, t]
        buf[:, 1] = p_t[:, 1, t]
        scat.set_offsets(buf)
        return [scat]

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, cache_frame_data=False)
    writer = PillowWriter(fps=max(1, int(1000 / interval)))
    anim.save(save_path, writer=writer)
    plt.close(fig)
