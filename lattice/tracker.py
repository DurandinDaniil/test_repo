from itertools import groupby
import operator
from typing import Tuple
from functools import reduce, partial
from collections import deque

import cv2

from .point_flow.grid_structure import Grid, get_frame_structure, stabilize_chains
from .point_flow.propagation import do_path_connection_step
from .point_flow.filtering import fit, linear, quadratic

import numpy as np


class Pool(deque):
    """ Section accumulator. @TODO: replace with 1-d stuff, cause this process is really 1d-process
    """
    shape = None
    ax = None

    def __init__(self, chain_stats: np.ndarray, size: int = 10, await_time: int = 5):
        """

        :param chain_stats:
        :param size:
        :param await_time:
        """
        super(Pool, self).__init__(maxlen=size)

        self.chain_stats = chain_stats

        self.orbits = None
        self.orbits_fit = []
        self.new_section = False
        self.await_time = await_time
        self.forget_counter = []
        self.last_section = -1
        self.tracker = []
        self.inertia = False
        self.error_distribution = []
        self.unit_measure_calculated = False
        self.m = []
        self.distances = []
        self.updated = []

    def get_current_points(self):
        last = len(self.tracker) if self.last_section == -1 else self.last_section + 1

        points = []
        for _, (tie, measure) in enumerate(zip(self.get_all_points()[:last], self.distances[:last])):
            if not len(tie):
                continue
            if len(measure) and self.unit_measure_calculated:
                points.append((tie, int(self.chain_stats[self.get_section_base_level(measure)][0]), _))
            else:
                points.append((tie, -1, _))

        return points

    def set_image_shape(self, shape: tuple, ax):
        """
        :param shape:
        :return:
        """
        self.shape = shape
        self.ax = ax

    def norm_to_base_level(self, x: np.ndarray, y: np.ndarray, bias: tuple = (0, 0),
                           weight: tuple = (1, 1)) -> np.ndarray:
        """

        :param x:  y-position / height of the section
        :param y:  distance to the nearest section above [in pixels]
        :param bias:
        :param weight:
        :return:
        """
        x0, y0 = bias
        vx, vy = weight
        # 142.12 ~ chain_stats[0, 1] ~ base level value
        return self.chain_stats[0, 1] * np.asarray(y) / ((np.asarray(x) - x0) * vy / vx + y0)

    def calculate_distances(self):  # chains
        """ Calculate distances between adjacent sectors.

        :param updated:
        :return:
        """
        if len(accumulated := self.get_all_points()) > 1:
            for index in [1]:
                if len(self.distances) and len(self.distances[0]):  # do it only for the first sleep
                    break

                if not len(self.tracker[0]):   # awaits of new section
                    break

                if not all(self.updated[:2]):  # otherwise coordinates are not real
                    break

                points = list(map(lambda _: np.asarray(_).mean(axis=0).reshape((2, 1)),
                                  [accumulated[_] for _ in [index, index - 1]]))
                self.distances[index - 1] = [(points[-1][-1][0], np.linalg.norm(operator.sub(*points)), 0, 0)]
                #
                # current_sector = [accumulated[_] for _ in [index, index - 1]]
                # if current_sector[0][0][1] < current_sector[-1][0][1]:
                #     upper_line = current_sector[0]
                #     lower_line = current_sector[1]
                # else:
                #     upper_line = current_sector[1]
                #     lower_line = current_sector[0]
                #
                # upper_eq = np.poly1d(np.polyfit(list(zip(*upper_line))[0], list(zip(*upper_line))[1], deg=1))
                # lower_eq = np.poly1d(np.polyfit(list(zip(*lower_line))[0], list(zip(*lower_line))[1], deg=1))
                # chain_dots = [x.tolist() for x in chains if (x[1] <= lower_eq(x[0])) and (x[1] >= upper_eq(x[0]))]
                #
                # chains_count = len(chain_dots) // 2
                # if chains_count in self.chain_stats[:, 0]:
                #     self.distances[index - 1].append((points[-1][-1][0], np.linalg.norm(operator.sub(*points)), chains_count))

    # def calculate_unit_measure_by_chains(self):
    #     if len(X := reduce(operator.add, self.distances, [])):
    #         x, y, *_ = zip(*X)
    #     # initial to convert # section to section length

    def calculate_unit_measure(self, threshold: int = 4):
        """

        :return:
        """

        if not self.unit_measure_calculated and len(
                X := reduce(operator.add, self.distances, [])) > threshold:  # use only first five sections
            x, y, *_ = zip(*X)

            mx, my = map(np.median, [x, y])
            X, Y = [], []
            for x0, y0 in zip(x, y):
                if abs(x0 / mx - 1) < 0.1 and abs(y0 / my - 1) < 0.1:
                    X.append(x0)
                    Y.append(y0)

            x, y = X, Y

            if len(x) < threshold:
                return

            vx, vy, x0, y0 = cv2.fitLine(
                points=np.asarray(list(zip(x, y)), dtype=np.float32),
                distType=cv2.DIST_WELSCH,
                param=0,
                reps=0.01,
                aeps=0.01)

            self.norm_to_base_level = partial(self.norm_to_base_level, bias=(x0, y0), weight=(vx, vy))
            self.unit_measure_calculated = True

    def displacement(self, base: list, to: list) -> list:
        return (np.asarray(to, dtype=np.float32) - np.asarray(base, dtype=np.float32)).mean(axis=0).tolist()

    def append(self, *args, **kwargs):
        """ Accumulate points history. FIFO queue -- append new to the first position and remove from the end position

        :param args:
        :param kwargs:
        :return:
        """
        grid, G, Gm, dZm = args  # g, G, Gm, dZm | grid -> G(grid)

        self.updated = [False] * len(self.tracker)  # remember all new point positions

        # accumulate complete structure from the historical data
        if not self.m:
            self.tracker = []
            for index, _ in enumerate(G):
                if _ != -1:
                    self.tracker.append([grid.points[i] if i != -1 else [np.inf] * 2 for i in grid.indexes[_]])
                    self.m.append(index)

            self.forget_counter = [0] * len(self.tracker)
            self.last_section = -1
            self.distances = [[] for _ in range(len(self.tracker))]  # store distances between adjacent section
            self.updated = [True] * (len(self.tracker))
            self.error_distribution = []
            self.orbits_fit = []
            self.unit_measure_calculated = False
        else:
            self.forget_counter = [operator.add(_, 1) for _ in self.forget_counter]
            for _, (e, f, mv) in enumerate(zip(G, Gm, dZm)):
                dy, dx = mv.mean(axis=0)
                if dy > 1:
                    print(_, dy, dx)  # @TODO: coherent |
                    continue
                if f == -1 and not _ and len(G) > 1 and G[_ + 1] > 0:
                    self.new_section = (self.displacement(
                        base=self.tracker[0], to=[grid.points[i] for i in grid.indexes[_]])[0] > 0)
                elif f != -1 and (self.displacement(self.tracker[f], [grid.points[i] for i in grid.indexes[e]])[0] < 0):
                    self.tracker[f] = [grid.points[i] for i in grid.indexes[e]]
                    self.updated[f] = True
                    self.forget_counter[f] = 0

            if self.new_section:
                self.tracker.insert(0, [grid.points[i] for i in grid.indexes[0]])
                self.new_section = False
                self.forget_counter = [0] + self.forget_counter
                self.updated = [False] + self.updated
                self.distances.insert(0, [])

            self.m = G

        # measure distances
        self.calculate_distances()

        # calculate unit-measure | calibration process
        self.calculate_unit_measure()

        if len(self.forget_counter) and self.forget_counter[self.last_section] >= self.await_time:
            self.last_section -= 1
            if len(self.forget_counter) < -self.last_section:
                self.m = []

        # complexification @TODO: flow restriction by stationary state
        # dZ = np.vectorize(complex)(dZ[..., 0], dZ[..., 1])

        if len(pts := np.asarray(self.get_all_points())) and not self.has_fit():
            if self.orbits is None:
                self.orbits = [[], [], [], []]

            [self.orbits[_].extend(pts[:, _, :].tolist()) for _ in range(4)]

            if all(self.orbits):
                try:
                    flags, fits, errors = zip(*[fit(_, yx=True) for _ in self.orbits])
                    if np.vstack(errors).reshape(-1, 1).max() < 0.5:  # @TODO: Do not use the constant, pre-define the parameter
                        print("fit on")
                        self.orbits_fit = fits
                except ValueError:
                    pass

        # append
        super(Pool, self).append(args)

    def get_indexes(self, max_value: int = np.iinfo('int').max, last_index: int = -1):
        """

        :param max_value: approx. value to inf
        :param last_index:
        :return:
        """

        last_index = len(self.tracker) if last_index == -1 else last_index + 1
        points = list(set(reduce(lambda x, y: x + y, self.tracker[:last_index], [])))
        indexes = [list(map(points.index, _)) if len(_) and u <= self.await_time else [-1, -1, -1, -1] for _, u in zip(self.tracker[:last_index], self.forget_counter[:last_index])]

        return points, indexes

    def has_fit(self) -> bool:
        if self.orbits_fit:  # @TODO: these conditions are not sufficient
            return True

        return False

    def reset(self):
        self.clear()
        self.orbits_fit = []
        self.orbits = None
        self.error_distribution = []
        self.unit_measure_calculated = False

    def fit(self, fs: Grid) -> Grid:
        """one-pass fit procedure

        :param fs:
        :return:
        """
        # firstly fit by `approximation fit`
        if fs.indexes:
            i = np.vstack(fs.indexes)
            p = fs.points

            points = [[p[_] for _ in orbit] for n, orbit in enumerate(i.T)]
            point_update = []  # fs.points
            for _, orbit in enumerate(points):
                x, y = zip(*orbit)
                a, b, c = self.orbits_fit[_]
                x_ = quadratic(np.asarray(y), a, b, c)

                point_update.extend([(fs.points.index(_), x_[i]) for i, _ in enumerate(orbit)])
                mean_error = sum((np.asarray(x) - x_) ** 2) / len(x)
                if mean_error > 5:  # @TODO: Do not use constant.
                    self.reset()
                    return fs
                # print("dx mean error", mean_error)

            for index, value in point_update:
                fs.points[index] = (value, fs.points[index][1])

            # linearize
            p = fs.points
            for index in fs.indexes:
                points = [p[_] for _ in index]
                flags, fits, errors = fit(points, func=linear)
                x, y = zip(*points)
                a, b = fits
                y_ = linear(np.asarray(x), a, b)
                point_update = [(fs.points.index(_), y_[j]) for j, _ in enumerate(points)]
                for index, value in point_update:
                    fs.points[index] = (fs.points[index][0], value)

        return fs

    def get_all_points(self, updated_only: bool = False):
        """

        :return:
        """
        last_index = len(self.tracker) if self.last_section == -1 else self.last_section + 1

        if updated_only:
            return [_[0] for _ in filter(lambda x: len(x[0]) and x[1], list(zip(self.tracker[:last_index], self.updated[:last_index])))]

        return list(filter(len, self.tracker[:last_index]))

    def get_section_base_level(self, section, percentage: float = 0.8):
        """

        :param section:
        :param percentage:
        :return:
        """
        level = -1
        x, y, *_ = list(zip(*section))

        indexes = list(map(
            lambda l: np.argmin(np.abs(self.chain_stats[:, 1] - l)),
            self.norm_to_base_level(x, y)))

        for (index, counts) in zip(*np.unique(indexes, return_counts=True)):
            if counts / len(indexes) > percentage:
                print(f"{index} = {self.chain_stats[index, 1]} | {counts / len(indexes)}")
                level = index
                break

        return level

    def get_filtered_sections(self, section, level, index, var=0.1):
        """

        :param section:
        :param level:
        :param index:
        :param var:
        :param d:
        :return:
        """
        sections = []
        starts, ends = [], []
        x, y, *_ = list(zip(*section))
        y_ = abs(self.norm_to_base_level(x, y) / self.chain_stats[level, 1] - 1)
        for s, y in list(zip(section, y_)):
            # if y <= var:
            start, end = s[-2:]
            starts.append(start)
            ends.append(end)
            sections.append([
                [list(self.tracker[index][0]), list(self.tracker[index][-1])],
                [list(self.tracker[index + 1][0]), list(self.tracker[index + 1][-1])],
                self.chain_stats[level][-1]
            ])

        return sections, starts, ends

    def get_all_sectors(self, filter_by_density: bool = False, threshold: float = 10.0):
        """

        :param min_count:
        :param percentage:
        :param filter_by_density:
        :param threshold:
        :return:
        """
        sectors = []

        if not self.unit_measure_calculated:
            return sectors
        # if not self.has_fit():
        #     return []

        # @TODO: add re-projection type / random, head, tail -- the best type is random subsample from all...
        # all_sectors = {}
        # transition = {}
        for i, section in enumerate(self.distances):
            if len(section):
                level = self.get_section_base_level(section)
                if level != -1 and i + 1 < len(self.tracker) and all(self.updated[i:i + 2]):
                    sectors.append([
                        [list(self.tracker[i][0]), list(self.tracker[i][-1])],
                        [list(self.tracker[i + 1][0]), list(self.tracker[i + 1][-1])],
                        self.chain_stats[level][-1]
                    ])
        #             filtered_sections, starts, ends = self.get_filtered_sections(section, level, i)
        #             for sector, s, e in zip(filtered_sections, starts, ends):
        #                 transition[(i, s)] = (i + 1, e)
        #                 all_sectors[(i, s)] = sector
        #             # sectors.extend(filtered_sections)
        #
        # values = list(set(list(transition.keys()) + list(transition.values())))
        # index_k = list(map(values.index, transition.keys()))
        # index_v = list(map(values.index, transition.values()))
        # # slices = []
        # for head in np.setdiff1d(index_k, index_v):
        #     slice = []
        #     q = values[head]
        #     slice.append(all_sectors[q])
        #     while q in transition:
        #         p = transition[q]
        #         if p not in all_sectors:
        #             break
        #         slice.append(all_sectors[p])
        #         q = p
        #
        #         return reduce(operator.add, sectors_grouped.values(), [])

        return sectors

    def set_predict_by_inertia(self, inertia: bool = False):
        """

        :param inertia:
        :return:
        """
        self.inertia = inertia

    def get_error_distribution(self):
        """

        :return:
        """
        d = {}
        if len(self.error_distribution):
            for key, group in groupby(self.error_distribution, lambda _: int(_[0])):
                d[key] = np.median(list(zip(*list(group)))[-1])
        return d


def mask2points(mask: np.ndarray, mode: str = 'peaks', base_level: int = 6) -> Grid:
    """U-Net mask to ordered grid.

    :param mask:
    :param mode:
    :param base_level:
    :return:

    """

    fs = get_frame_structure(mask, base_level=base_level, mode=mode)
    return stabilize_chains(fs)


def points2points(m_fs: Grid, c_fs: Grid) -> Tuple[list, np.ndarray]:  # , dict]:
    """Find correspondence between two consequent frames

    :param m_fs: previous (mean) frame
    :param c_fs: current frame
    :return:
    """

    # orbits = {0: [], 1: [], 2: [], 3: [], 'mean': []}
    # new structure for new frame
    if m_fs.indexes and c_fs.indexes:  # otherwise it's required to restore all chains
        # neighbour_arg -> indexes of the nearest points in `m_points` to `frame_points`
        # neighbour_dst -> their distances
        neighbour_arg, neighbour_dst = do_path_connection_step(
            p=(np.asarray(c_fs.indexes), c_fs.points), q=(np.asarray(m_fs.indexes), m_fs.points))

        # mark non-minimal copies : #(copies) > 1 => index of each copy = -1
        for value, _ in filter(lambda _: _[-1] > 1, zip(*np.unique(neighbour_arg, return_counts=True))):
            row, col = np.where(neighbour_arg == value)
            minimal_arg = np.argmin(neighbour_dst[(row, col)])
            minimal_index = (row[minimal_arg], col[minimal_arg])
            minimal_value = neighbour_arg[minimal_index]

            neighbour_arg[(row, col)] = -1
            neighbour_arg[minimal_index] = minimal_value

        dz = np.zeros(shape=(*neighbour_arg.shape, 2))
        for _, (start, end) in enumerate(zip(neighbour_arg, c_fs.indexes)):
            if -1 in start:
                continue
            start, end = [np.asarray(points)[_] for points, _ in [(m_fs.points, start), (c_fs.points, end)]]
            dz[_] = (end.astype(np.float32) - start.astype(np.float32)).astype(np.int)

            # # mean
            # [orbits[i].append((*start[i], *dz[_][i])) for i in range(len(dz[_]))]
            # orbits['mean'].append((np.mean(start[:, 0]), np.mean(start[:,1]), np.mean(dz[_][:,0]), np.mean(dz[_][:,1])))

        indexes = []
        for _ in neighbour_arg:
            if -1 in _:
                indexes.append(-1)
            # elif list(_) in m_fs.indexes:
            elif list(_) in m_fs.indexes:
                indexes.append(m_fs.indexes.index(list(_)))
            # else:
            #     for i, index in enumerate(m_fs.indexes):
            #         if len(set(index) | set(_)) < 6:  # @FixMe -- differ only by one [or two elements | order]
            #             indexes.append(i)
            #             break

        return indexes, dz  # , orbits

    return [], np.asarray([])  # , orbits


def form_grid(points: list, indexes: list) -> Grid:
    return Grid(
        points=points,
        anchor=[],
        t=[],
        indexes=indexes,
        ordered_chains=[],
        encoded_chains=None,
        fit_chains=[]
    )
