from functools import reduce, partial

import cv2
import numpy as np
from photutils.detection import find_peaks


def posterize(image: np.ndarray, levels: float = 3.0) -> np.ndarray:
    """Image-2-Image

    :param image:
    :param levels:
    :return:
    """
    image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    gray = 255 * np.round((image / 255.0) * levels) / levels

    return gray.astype(np.uint8)


def threshold(image: np.ndarray, min_val=254, max_val=255) -> np.ndarray:
    """Image-2-Image

    :param image:
    :param min_val:
    :param max_val:
    :return:
    """

    _, th = cv2.threshold(image, min_val, max_val, cv2.THRESH_BINARY)
    return th


def get_centers(image: np.ndarray, mode: str = 'centroid', levels: float = 4) -> list:
    """Image-2-Points

    :param levels:
    :param image:
    :param mode: possible values are 'peaks', 'centroids', 'max'
    :return:
    """

    if mode == 'peaks':
        points = find_peaks(image, threshold=0.5)  # @TODO: box_
        if points is None:
            dots = np.array([[0, 0]])
        else:
            dots = np.array([list(j) for j in points], dtype=np.uint32)[:, :2]

        return [dots[1:, 0], dots[1:, 1]]

    th = reduce(lambda x, y: y(x), [image, partial(posterize, levels=levels), threshold])
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8, cv2.CV_32S)

    if mode == 'centroid':
        centers = []

        for i in range(1, n):
            component_mask = (labels == i).astype("uint8") * 255
            *_, max_loc = cv2.minMaxLoc(image, mask=component_mask[:, :])
            centers.append(max_loc)

        return list(zip(*centers))

    if mode == 'max':
        return [centroids[1:, 0], centroids[1:, 1]]

    return []
