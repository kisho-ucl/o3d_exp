import open3d as o3d
import numpy as np
import copy
import pandas as pd
from open3d_fast_global_registration import execute_fast_global_registration


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def filter_pcd(pcd1, pcd2, z_threshold):
    # ICPの適用
    #z_threshold = vc[2]+0.05  # Replace with your desired threshold
    points = np.asarray(pcd1.points)
    colors = np.asarray(pcd1.colors)
    mask = points[:, 2] >= z_threshold
    # Filter points: keep only points where z > z_threshold
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    # Create a new point cloud with filtered points
    filtered_pcd1 = o3d.geometry.PointCloud()
    filtered_pcd1.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd1.colors = o3d.utility.Vector3dVector(filtered_colors)

    points = np.asarray(pcd2.points)
    colors = np.asarray(pcd2.colors)
    mask = points[:, 2] >= z_threshold
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    # Create a new point cloud with filtered points
    filtered_pcd2 = o3d.geometry.PointCloud()
    filtered_pcd2.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd2.colors = o3d.utility.Vector3dVector(filtered_colors)

    source = filtered_pcd1 #.voxel_down_sample(0.002)
    target = filtered_pcd2 #.voxel_down_sample(0.002)
    return filtered_pcd1, filtered_pcd2

                                    

# Load point clouds
n=4
id_src = 1
id_tgt = id_src - 1

source = o3d.io.read_point_cloud(f"./datasets{n}/pcd_{id_src}.ply")
camera_pose_path_src = f"./datasets{n}/camera_pose_{id_src}.txt"
camera_pose_df_src = pd.read_csv(camera_pose_path_src, header=None)
camera_pose_src = camera_pose_df_src.values
trans_src = np.linalg.inv(camera_pose_src)
source.transform(camera_pose_src)

target = o3d.io.read_point_cloud(f"./datasets{n}/pcd_{id_tgt}.ply")
camera_pose_path_tgt = f"./datasets{n}/camera_pose_{id_tgt}.txt"
camera_pose_df_tgt = pd.read_csv(camera_pose_path_tgt, header=None)
camera_pose_tgt = camera_pose_df_tgt.values
trans_tgt = np.linalg.inv(camera_pose_tgt)
target.transform(camera_pose_tgt)
target_copy = target
initial_trans = np.identity(4)
print(initial_trans)


source2, target2 = filter_pcd(source, target,0.0)
# ダウンサンプリング (ボクセルサイズを調整)
voxel_size = 0.01
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)
o3d.visualization.draw_geometries([source2, target2])
# 法線の推定
source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#o3d.visualization.draw_geometries([source_down, target_down])

def compute_fpfh(point_cloud, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # 法線の推定
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH特徴記述子の計算
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh

import open3d as o3d

def compute_fpfh_fast(point_cloud, voxel_size):
    # 法線推定の半径と特徴記述子計算の半径を調整
    radius_normal = voxel_size * 1.5  # 法線計算の半径 (小さめに設定)
    radius_feature = voxel_size * 3   # FPFH特徴記述子の計算半径 (小さめに設定)

    # 法線の推定
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))

    # FPFH特徴記述子の計算
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        point_cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50)
    )
    return fpfh

# 点群の読み込みとダウンサンプリング
voxel_size = 0.1  # 大きめのボクセルサイズで高速化
#source = o3d.io.read_point_cloud("source.pcd").voxel_down_sample(voxel_size)


source_fpfh = compute_fpfh(source_down, voxel_size)
target_fpfh = compute_fpfh(target_down, voxel_size)

# RANSACによる初期アライメント
distance_threshold = voxel_size * 1.5
"""
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, True,
    distance_threshold,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    3,  # RANSACの反復数
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)
"""
fgr_option = o3d.pipelines.registration.FastGlobalRegistrationOption()
result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh, fgr_option
)
print(result.transformation)

# ICPによる精密アライメント
#result_icp = o3d.pipelines.registration.registration_icp(
#    source_down, target_down, distance_threshold, result_ransac.transformation,
#    o3d.pipelines.registration.TransformationEstimationPointToPlane()
#)

#print(result_icp)
#source.paint_uniform_color([1, 0, 0])  # 赤色
#target.paint_uniform_color([0, 1, 0])  # 緑色
source.transform(result.transformation)
# 結果の可視化
o3d.visualization.draw_geometries([target,source])
