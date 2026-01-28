"""
Utility functions for sythetic data. Mostly plotting functions
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter


def make_state_vars(pos, vel, edges, dt, save_fp=None):
    """
    makes a dict of state variables
    
    :param pos: posiition vectors
    :param vel: velocity vectors
    :param edges: edges either over time or just ground truth edges
    """
    state_vars = {
        'x': np.array(pos),
        'x_dot': np.array(vel),
        'x_dot_dot': np.gradient(vel, dt, axis=-1),
        'edges': edges
    }
    return state_vars


def plot_trajectories(p_t, save_path, A=None, arena_radius=1.0, force_field_radius=None, step=10):
    # p_t: (2, n, T), A: (n, n)
    n = p_t.shape[1]
    T = p_t.shape[2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle((0, 0), arena_radius, edgecolor='black', facecolor='none', lw=1.5))
    if force_field_radius is not None:
        ax.add_patch(Circle((0, 0), force_field_radius, edgecolor='orange', facecolor='none', ls='--', lw=1))

    for i in range(n):
        ax.plot(p_t[0, i, ::step], p_t[1, i, ::step], lw=1)
        ax.scatter(p_t[0, i, -1], p_t[1, i, -1], s=25)

    # draw edges at final frame (if provided)
    if A is not None:
        ii, jj = np.where(np.triu(A, 1))
        x = p_t[0, :, -1]
        y = p_t[1, :, -1]
        for k in range(len(ii)):
            ax.plot([x[ii[k]], x[jj[k]]], [y[ii[k]], y[jj[k]]], color='gray', alpha=0.4, lw=0.8)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-arena_radius * 1.05, arena_radius * 1.05)
    ax.set_ylim(-arena_radius * 1.05, arena_radius * 1.05)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spring–mass trajectories')
    plt.savefig(save_path)


def animate_system(p_t, A_t, save_path, arena_radius=1.0, force_field_radius=None,
start=0, stop=None, step=100, interval=30):
    """
    p_t: (n, 2, T), A_t: (n, n, T) boolean
    """
    n = p_t.shape[0]
    T = p_t.shape[2]
    stop = T if stop is None else min(stop, T)
    frames = range(start, stop, step)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Circle((0, 0), arena_radius, edgecolor='black', facecolor='none', lw=1.5))
    if force_field_radius is not None:
        ax.add_patch(Circle((0, 0), force_field_radius, edgecolor='orange', facecolor='none', ls='--', lw=1))

    scat = ax.scatter(p_t[:, 0, start], p_t[:, 1, start], s=30, c=np.arange(n), cmap='tab10')
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    lines = [ax.plot([], [], color='gray', alpha=0.5, lw=0.8)[0] for _ in pairs]

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-arena_radius * 1.05, arena_radius * 1.05)
    ax.set_ylim(-arena_radius * 1.05, arena_radius * 1.05)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Spring–mass with dynamic edges')

    buf = np.empty((n, 2), dtype=p_t.dtype)

    def update(t):
        buf[:, 0] = p_t[:, 0, t]
        buf[:, 1] = p_t[:, 1, t]
        scat.set_offsets(buf)
        if A_t is not None:
            A = A_t[:, :, t].astype(bool)
            for k, (i, j) in enumerate(pairs):
                if A[i, j]:
                    lines[k].set_data([buf[i, 0], buf[j, 0]], [buf[i, 1], buf[j, 1]])
                    lines[k].set_visible(True)
                else:
                    lines[k].set_visible(False)
        return [scat] + lines

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True, cache_frame_data=False)

    # GIF via PillowWriter (no ffmpeg). Downsample with step to avoid OOM.
    writer = PillowWriter(fps=max(1, int(1000 / interval)))
    anim.save(save_path, writer=writer)
    plt.close(fig)
