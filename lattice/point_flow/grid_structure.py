from collections import namedtuple
# from functools import partial, reduce

import numpy as np
from scipy.spatial.distance import euclidean

from .segmentation import get_centers
from .triangulation import get_simplicial_complex, get_base_point, get_chain, order_chain
from .triangulation import QError

# @TODO: update to `recordtype`
Grid = namedtuple('Grid', [
    'points',
    # 'recovery_points',
    'anchor',
    't',
    'indexes',
    'ordered_chains',
    'encoded_chains',
    'fit_chains'
])


def get_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    :param a:
    :param b:
    :return:
    """

    _, ncols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [a.dtype]}

    return np.setdiff1d(a.view(dtype), b.view(dtype)).view(a.dtype).reshape(-1, ncols)


def get_points_at_scale(frame: np.ndarray, levels: float = 4, mode: str = 'centroid') -> list:
    """

    :param frame:
    :param levels:
    :param mode:
    :return:
    """
    # get base points / get additional point for recovering structure
    # points = []  # should be non-empty!
    # (3, 4, 5), (3, ), (5,), (3, 5), etc

    return list(zip(*get_centers(frame, mode=mode, levels=levels)))


def get_frame_structure(frame: np.ndarray, mode: str = 'centroid', base_level: float = 5) -> Grid:
    """Get per-frame structure of grid.

    :param angle_threshold:
    :param mode:
    :param base_level:
    :param frame:
    :return:
    """

    assert base_level > 2
    points = get_points_at_scale(frame, levels=base_level, mode=mode)

    if not len(points):
        return Grid(
            points=points,
            anchor=[],
            t=None,
            indexes=None,
            ordered_chains=[],
            encoded_chains=None,
            fit_chains=[]
        )

    try:
        t_chain_base = get_simplicial_complex(points)

        anchor_point, angle_mean = get_base_point(t_chain_base)
        chain = get_chain(anchor_point, t_chain_base.points, angle_mean=angle_mean)
        ordered = order_chain(chain)[::-1]  # reverse order

        # 0 -  filter out outliers / by distance, by angle / by chain intersection
        min_distance = 1 / 2.
        for i in range(len(ordered)):
            chain = ordered[i]
            distances = [euclidean(t_chain_base.points[u], t_chain_base.points[v]) for (u, v) in
                         zip(chain[:-1], chain[1:])]
            m = np.median(distances)  # np.mean(distances)
            n = np.asarray([m] + distances) / m
            indexes = np.where((n > min_distance))
            ordered[i] = np.asarray(chain)[indexes].tolist()

            if n[1] > 3:  # start [3 times bigger then median distance between points in the chain]
                ordered[i].pop(0)
            if n[-1] > 3:  # end
                ordered[i].pop()

        # 1 -  ...
        return Grid(
            points=points,
            # recovery_points=recovery_points,
            anchor=anchor_point,
            t=t_chain_base,
            indexes=None,
            ordered_chains=ordered,
            encoded_chains=None,
            fit_chains=[]
        )
    except QError:
        return Grid(
            points=points,
            # recovery_points=recovery_points,
            anchor=[],
            t=None,
            indexes=None,
            ordered_chains=[],
            encoded_chains=None,
            fit_chains=[]
        )


def stabilize_chains(current: Grid, mode='filter') -> Grid:
    """

    :param current:
    :param mode: possible modes are "filter", "per-frame-recovery", "merge", or any their combination
    :param past:
    :return:
    """

    # modes = mode.split("+")

    if mode == 'filter':
        # by length
        pre_index = list(filter(lambda x: len(x) == 4, current.ordered_chains))

        # if coincide
        indexes = []
        skip = -1
        for i, (a, b) in enumerate(zip(pre_index[0:], pre_index[1:])):
            if i == skip:
                continue
            if len(np.unique(a + b)) == len(a) + len(b):
                indexes.append(a)
            else:
                skip = i + 1

        if skip != len(pre_index) - 2 and len(pre_index):
            indexes.append(pre_index[-1])

        current = current._replace(indexes=indexes)

    return current
