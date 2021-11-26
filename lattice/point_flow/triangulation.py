"""Rigid structure.
"""

from typing import Tuple
from itertools import permutations

import numpy as np
import scipy.spatial.qhull
from scipy.spatial.qhull import QhullError as QError
from scipy.spatial import Delaunay

from sklearn.mixture import GaussianMixture


def get_simplicial_complex(points: list) -> scipy.spatial.qhull.Delaunay:
    """Do Delaunay's plane triangulation

    :param points: non empty list of N-points on the plane [(x_1,y_1), ..., (x_N, y_N)]
    :return: scipy.spatial.qhull.Delaunay
    """

    return Delaunay(points, incremental=True)


def get_base_angle(t: scipy.spatial.qhull.Delaunay) -> float:
    """Get minimal angle.

    :param max_angle:
    :param min_angle:
    :param t:
    :return:
    """

    angle_distribution = []
    for i, simplex in enumerate(t.simplices):
        for _ in range(3):
            v = t.points[simplex[(_ + 1) % len(simplex)]] - t.points[simplex[_]]
            w = t.points[simplex[_ - 1]] - t.points[simplex[_]]

            # angle_distribution.extend([np.abs(angle(w)), np.abs(angle(v))])
            angle_distribution.extend([angle(w), angle(v)])

    return get_minimal_angle(angle_distribution)


def get_base_point(t: scipy.spatial.qhull.Delaunay, max_angle: float = 25.) -> Tuple[list, float]:
    """Order simplex points according to `fixed` coordinate orientation

    :param max_angle:
    :param t:
    :return:
    """

    base_angle = get_base_angle(t)
    base_points = []
    for i, simplex in enumerate(t.simplices):
        for _ in permutations(range(3)):
            base_, right_, up_ = (simplex[_] for _ in _)
            v = angle(t.points[right_] - t.points[base_]) - base_angle
            if np.abs(v) < max_angle:
                base, right, up = base_, right_, up_
                base_points.append([base, right, up])

    return base_points, base_angle


def get_minimal_angle(angle_distribution: list, n_components: int = 3, std_threshold: float = 10.) -> float:
    """Overhead method of dividing the angle space into almost constant components.

    :param std_threshold:
    :param angle_distribution:
    :param n_components:
    :return:
    """

    # if n_components > 10:
    #     # n_components > 10 is extra rare, avg is ~7-8 (same as before fix)
    #     logger.info(f"n_components for GaussianMixture: {n_components}")

    X = np.asarray(angle_distribution)[:, np.newaxis]

    gmm = GaussianMixture(n_components, n_init=n_components, covariance_type='spherical', max_iter=500)
    gmm.fit(X)
    labels = gmm.predict(X)

    medians, std = [], []
    for j in range(n_components):
        indexes = np.where(labels == j)
        data = X[indexes]

        # np.nan sometimes appears here, later it is considered as min value, and then -> chain == [[]]
        median_angle = np.median(data)
        if not np.isnan(median_angle):
            medians.append(median_angle)
            std.append(np.std(data))

    if gmm.aic(X) < std_threshold ** 2:
        return float(medians[np.argmin([_ ** 2 for _ in medians])])

    # fixme: magic constant 50 deg. "valid" base_angles == ~0-18 deg
    if min([abs(i) for i in medians]) > 50:
        n_components += 1
        m = get_minimal_angle(angle_distribution, n_components)
        return m

    arg_min = np.argmin([_ ** 2 for _ in medians])
    if std[arg_min] > std_threshold:  #
        m = get_minimal_angle(angle_distribution, n_components + 1)
        return m

    return float(medians[arg_min])


def angle(p: np.array) -> float:
    """Get angle ∠poX

    :param p:
    :return:
    """
    x, y = p
    return np.angle([x + 1j * y], deg=True)[0]


def get_chain(simplices: list, points: np.ndarray, angle_mean: float = 0.) -> list:
    """Getting the list of 4-chains from the set of anchors.

    :param simplices: the list of simplices -- each encodes by `point` indexes
    :param points: set of points
    :param angle_mean: estimated angle of typical chain
    :return:
    """
    grid = dict()

    for i, simplex in enumerate(simplices):
        base, right, up = simplex
        rights = [right] + [_[1] for _ in filter(len, [s if s[0] == base and s[1] != right else [] for s in simplices])]

        base_angles = np.asarray([np.abs(angle(points[i] - points[base]) - angle_mean) for i in rights])
        distances = np.asarray([scipy.spatial.distance.euclidean(points[i], points[base]) for i in rights])  # [j]
        if base_angles.size:
            _ = np.argmin(distances)

            grid[base] = rights[_]

    return list(zip(grid.keys(), grid.values()))


def order_chain(chain: list):  # -> np.ndarray:
    """Encode list of chains.

    :param chain:
    :return:
    """
    ordered = []

    d = dict(chain)

    start = np.setdiff1d(list(d.keys()), list(d.values()))

    for i, start in enumerate(start):
        key = start
        ordered.append([key])
        while key in d:
            ordered[i].append(d[key])
            key = d[key]

    return ordered  # return np.vstack(ordered)
