import numpy as np
import matplotlib.pyplot as plt

import os
import sys
# Find path to the BERNAISE root folder
bernaise_path = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-2])
# ...and append it to sys.path to get functionality from BERNAISE
sys.path.append(bernaise_path)
from common import parse_command_line, info, info_blue, info_red, info_on_red
from generate_mesh import round_trip_connect, MESHES_DIR
from plot import plot_edges

from skimage import measure
from scipy import ndimage, misc


def main():
    cmd_kwargs = parse_command_line()

    image_path = cmd_kwargs.get("image", False)

    name = os.path.splitext(os.path.basename(image_path))[0]
    print name

    if not image_path or not os.path.exists(image_path):
        info_on_red("Image does not exist.")
        exit()

    image = misc.imread(image_path)
    image = np.array(np.array(np.mean(image[:, :, :3], 2),
                              dtype=int)/255, dtype=float)

    contours = measure.find_contours(image, 0.5)

    nodes = contours[0][::10, 1::-1]
    nodes /= np.max(nodes)
    nodes[:, 1] = -nodes[:, 1]

    nodes_max = np.max(nodes, 0)
    nodes_min = np.min(nodes, 0)
    nodes[:, 1] -= nodes_min[1]
    nodes[:, 0] -= nodes_min[0]

    edges = round_trip_connect(0, len(nodes)-1)

    savefile_prefix = os.path.join(MESHES_DIR, name)
    np.savetxt(savefile_prefix + ".nodes", nodes)
    np.savetxt(savefile_prefix + ".edges", edges, fmt='%i')

    plot_edges(nodes, edges)


if __name__ == "__main__":
    main()
