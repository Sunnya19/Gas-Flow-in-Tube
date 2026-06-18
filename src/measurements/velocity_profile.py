from typing import Tuple
import numpy as np


def compute_velocity_profile(
    positions: np.ndarray,
    velocities: np.ndarray,
    height: float,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the streamwise velocity profile u_x(y).

    Args:
        positions: (N,2) array of particle positions
        velocities: (N,2) array of particle velocities
        height: channel height H
        n_bins: number of bins across y

    Returns:
        y_centers: (n_bins,) array of bin center y coordinates
        ux_profile: (n_bins,) array of mean u_x in each bin (np.nan if empty)
    """
    bins = np.linspace(0.0, height, n_bins + 1)
    # bin centers
    y_centers = 0.5 * (bins[:-1] + bins[1:])

    sum_vx = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    # assign each particle to a bin
    y = positions[:, 1]
    vx = velocities[:, 0]
    # use digitize to get bin indices in 1..n_bins
    inds = np.digitize(y, bins) - 1

    # clamp indices (particles exactly on upper bound may give n_bins)
    inds[inds < 0] = -1
    inds[inds >= n_bins] = -1

    for i, idx in enumerate(inds):
        if idx == -1:
            continue
        sum_vx[idx] += vx[i]
        counts[idx] += 1

    ux_profile = np.full(n_bins, np.nan, dtype=float)
    nonzero = counts > 0
    ux_profile[nonzero] = sum_vx[nonzero] / counts[nonzero]

    return y_centers, ux_profile
