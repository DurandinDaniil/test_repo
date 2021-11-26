from pathlib import Path


import click
import cv2
import matplotlib.pyplot as plt

from lattice.tracker import mask2points, points2points, Pool
from lattice.point_flow.filtering import fit, objective


@click.command()
@click.option('--directory', default="", help='path to the image directory')
@click.option('--extension', default="png", help='image extension like `jpeg`, `png`, or so.')
def frame_processor(directory, extension):
    """toy stream of frames

    :param directory:
    :param extension:
    :return:
    """

    # ch_counter
    # https://gitlab.ntrlab.ru/videoai.ntr/evraz-lava/workers/homography/blob/master/homography/camera/chain_counter.py#L91
    pool = Pool(chain_stats=ch_counter.chain_stats, size=10)
    for i, p in enumerate(sorted(map(str, Path(directory).rglob(f'*.{extension}')))):
        print(i, p)
        dots_frame = cv2.imread(p)

        g = mask2points(dots_frame, mode="peaks")
        # chain_dots = np.array(list(zip(*segmentation.get_centers(chain_mask, mode="peaks"))))

        if self.grid_pool.has_fit():
            g = self.grid_pool.fit(fs=g)

        if len(g.indexes):
            self.grid.append(g)

        if len(G := list(range(len(g.indexes)))):
            # frame-by-frame
            # G, dZ = points2points(m_fs=self.grid[-1], c_fs=g)  # (previous, current)

            # frame-by-mean
            points, indexes = self.grid_pool.get_indexes(last_index=self.grid_pool.last_section)
            Gm, dZm = points2points(m_fs=form_grid(points=points, indexes=indexes), c_fs=g)

            self.grid_pool.append(self.grid[-1], G, Gm, dZm)


if __name__ == "__main__":
    frame_processor()
