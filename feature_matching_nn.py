#! usr/bin/python3

import copy
import time
import numpy as np
import open3d as o3d

from utils.utils_rot_trans import find_rotation_translation
from utils.utils_superpoint import find_matching_features_nn
from utils.utils import convert2transparent, correspondance_to_3dpoints

EXP_IDS = [30, 31]
CAPTURE_FOLDER = "cap_data"
folder = 'masked_imgs/exps'

convert2transparent(EXP_IDS, folder)

first_depth_data = np.loadtxt(f"{CAPTURE_FOLDER}/exp{EXP_IDS[0]}/depth_data.txt", delimiter=",")
second_depth_data = np.loadtxt(f"{CAPTURE_FOLDER}/exp{EXP_IDS[1]}/depth_data.txt", delimiter=",")

first_segmented_pcd = o3d.io.read_point_cloud(f"{CAPTURE_FOLDER}/exp{EXP_IDS[0]}/segmented_scene.pcd")
second_segmented_pcd = o3d.io.read_point_cloud(f"{CAPTURE_FOLDER}/exp{EXP_IDS[1]}/segmented_scene.pcd")

list_kp1, list_kp2 = find_matching_features_nn(EXP_IDS[0], EXP_IDS[1], folder)

first_correspondances_depth_data, second_correspondances_depth_data = [], []
for i in range(len(list_kp1)):
    first_correspondances_depth_data.append(first_depth_data[int(list_kp1[i][1]), int(list_kp1[i][0])])
    second_correspondances_depth_data.append(second_depth_data[int(list_kp2[i][1]), int(list_kp2[i][0])])

# Get from camera
CX_DEPTH, CY_DEPTH = 160.19308471679688, 118.8604736328125
FX_DEPTH, FY_DEPTH = 193.3643035888672, 193.3643035888672

first_pointset = correspondance_to_3dpoints(first_correspondances_depth_data, list_kp1, 
                                            CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH)
second_pointset = correspondance_to_3dpoints(second_correspondances_depth_data, list_kp2, 
                                             CX_DEPTH, CY_DEPTH, FX_DEPTH, FY_DEPTH)

start_transformation_time = time.time()
rot_mat, trans_mat = find_rotation_translation(first_pointset, second_pointset, weights=0)

# Create transformation matrix
T = np.eye(4)
T[:3, :3] = np.array(rot_mat)
T[:3, 3] = np.array(trans_mat).T # T[0, 3] = trans_mat[0][0]; T[1, 3] = trans_mat[0][1]; T[2, 3] = trans_mat[0][2]
end_transformation_time = time.time()

print(f">> T = {T}")
print(f">> Transformation time: {1000*(end_transformation_time - start_transformation_time):.4f} ms.")

aligned_pcd = second_segmented_pcd + copy.deepcopy(first_segmented_pcd).transform(T)
o3d.visualization.draw_geometries([aligned_pcd])
# o3d.io.write_point_cloud(filename=f"results/lab/aligned_lab_scenes.pcd", 
#                          pointcloud=aligned_pcd, write_ascii=True)
