import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# npyファイルの読み込み
points = np.load('./o3d_exp/info/c0_plane.npy')  # 'points.npy'を読み込む
points2 = np.load('./o3d_exp/info/table_pose.npy') 

# 3次元の円フィッティング用の関数
def circle_fit_cost(params, points):
    center = params[:3]  # 中心座標 [a, b, c]
    radius = params[3]   # 半径 r
    normal = params[4:]  # 平面の法線ベクトル [nx, ny, nz]
    normal /= np.linalg.norm(normal)  # 正規化

    # 各点から平面までの距離がほぼ0（平面上にある）ことを確認
    plane_distances = np.dot(points - center, normal)

    # 各点から円の中心までの距離が半径に近いことを確認
    circle_distances = np.linalg.norm((points - center) - plane_distances[:, None] * normal, axis=1) - radius

    # 平面への適合度と円への適合度を合計
    return np.sum(plane_distances**2) + np.sum(circle_distances**2)


def get_table_pose_from_points(points):
    # 初期推定値（平均点を中心、初期半径1、適当な法線）
    initial_guess = np.append(np.mean(points, axis=0), [1.0, 0, 0, 1])

    # 最適化によって円の中心、半径、法線を探す
    result = minimize(circle_fit_cost, initial_guess, args=(points,))
    center_point = result.x[:3]  # 中心
    radius = result.x[3]         # 半径
    normal_vector = result.x[4:] / np.linalg.norm(result.x[4:])  # 法線ベクトルを正規化
    z_axis = normal_vector

    return center_point, radius, normal_vector

def orthogonal_vectors(normal):
    # 適当なベクトル (1, 0, 0) が法線ベクトルに直交するか確認
    if np.allclose(normal, [1, 0, 0]) or np.allclose(normal, [-1, 0, 0]):
        v1 = np.array([0, 1, 0])
    else:
        v1 = np.array([1, 0, 0])
    
    v2 = np.cross(normal, v1)
    v1 = np.cross(normal, v2)
    
    return v1, v2

center_point, radius, normal_vector = get_table_pose_from_points(points)
z_axis = normal_vector
x_axis, y_axis = orthogonal_vectors(normal_vector)

# 3Dプロットの設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 点群のプロット
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='g', marker='o')

# 中心点のプロット
ax.scatter(center_point[0], center_point[1], center_point[2], c='r', marker='x', s=100, label='Center Point')

# 回転軸（法線ベクトル）の表示
rotation_axis_length = 1.0  # 回転軸の長さを設定
rotation_start = center_point
rotation_end = rotation_start + normal_vector * rotation_axis_length

ax.quiver(rotation_start[0], rotation_start[1], rotation_start[2], 
          normal_vector[0], normal_vector[1], normal_vector[2], 
          length=rotation_axis_length, color='g', label='Rotation Axis')

# 円の弧の可視化（3次元円の一部を表示）
theta = np.linspace(0, 2 * np.pi, 100)
circle_points = np.array([center_point + radius * (np.cos(t) * np.cross(normal_vector, [1, 0, 0]) + 
                                                    np.sin(t) * np.cross(normal_vector, np.cross(normal_vector, [1, 0, 0]))) 
                          for t in theta])
ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='cyan', label='Fitted Circle')



# x 軸と y 軸の表示
axis_length = 1.0  # x 軸と y 軸の長さを設定
ax.quiver(center_point[0], center_point[1], center_point[2], 
          x_axis[0], x_axis[1], x_axis[2], 
          length=axis_length, color='b', label='X Axis')

ax.quiver(center_point[0], center_point[1], center_point[2], 
          y_axis[0], y_axis[1], y_axis[2], 
          length=axis_length, color='m', label='Y Axis')

# ラベルの設定
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim([-0.2,0.2])
ax.set_ylim([-1.0,-0.6])
ax.set_zlim([0,1])
ax.legend()

# グラフの表示
plt.show()

print("円の中心:", center_point)
print("半径:", radius)
print("回転軸（法線ベクトル）:", normal_vector)
