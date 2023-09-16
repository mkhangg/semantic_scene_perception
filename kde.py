#! usr/bin/python3

import time
import numpy as np
import open3d as o3d

from KDEpy import FFTKDE
from utils.utils import extract_xyz_coordinates, get_probs_by_dimension


pcd_file_path = "exp_feature/first_correspondance_pcd.pcd"
correspondance_pcd = o3d.io.read_point_cloud(filename=pcd_file_path)
xs, ys, zs = extract_xyz_coordinates(correspondance_pcd)

start_weighing_time = time.time()
xx, yx = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(xs).evaluate()
end_weighing_time = time.time()

xy, yy = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(ys).evaluate()
xz, yz = FFTKDE(kernel='gaussian', bw='ISJ', norm=1).fit(zs).evaluate()
print(f">> Weighing time: {1000*(end_weighing_time - start_weighing_time):.4f} ms.")

probs_x = get_probs_by_dimension(xs, xx, yx)
probs_y = get_probs_by_dimension(ys, xy, yy)
probs_z = get_probs_by_dimension(zs, xz, yz)
probs = np.array((probs_x * probs_y * probs_z))
