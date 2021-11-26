
from typing import Tuple

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def do_path_connection_step(p: Tuple[np.ndarray, list], q: Tuple[np.ndarray, list], distance='l1') -> Tuple[np.ndarray, np.ndarray]:
    """Form locally `arc-wise connected` space

    :return:
    """

    x, y = [[p_[_.tolist()] for _ in np.nditer(i_)] for i_, p_ in [p, q]]

    neighbour_arg, neighbour_dst = pairwise_distances_argmin_min(X=x, Y=y, metric=distance)

    func_vec = np.vectorize(lambda _: q[-1].index(y[_]))  # re-coordinate

    return func_vec(neighbour_arg).reshape(p[0].shape), neighbour_dst.reshape(p[0].shape)
