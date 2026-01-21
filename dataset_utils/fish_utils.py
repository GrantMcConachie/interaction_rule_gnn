"""
Utility functions specifically for making fish graphs
"""

import numpy as np
import pandas as pd


def generate_x_dot(loc, frame_rate=120):
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
        vel = np.gradient(pos, axis=1) * frame_rate
        acc = np.gradient(vel, axis=1) * frame_rate

        x.append(pos)
        x_dot.append(vel)
        x_dot_dot.append(acc)

    return {
        'x': np.array(x),
        'x_dot': np.array(x_dot),
        'x_dot_dot': np.array(x_dot_dot)
    }
