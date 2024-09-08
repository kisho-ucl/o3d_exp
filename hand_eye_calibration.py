import cv2
import numpy as np
from scipy.optimize import minimize

dir0 = f'/Users/kisho/open3d/o3d_exp/datas'

# 例としてランダムな変換行列のセットを作成します
def create_random_transformation():
    R, _ = cv2.Rodrigues(np.random.randn(3))  # ランダムな回転行列
    t = np.random.randn(3, 1)                 # ランダムな並進ベクトル
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

# カメラからマーカーへの変換行列 A_i
A = []
for i in range(15):
    T_cam2marker = np.load(f"{dir0}/T_cam2marker_{i}.npy")
    A.append(T_cam2marker)

# マップかエンドエフェクタへの変換行列 B_i
B = []
for i in range(15):
    T_map2end = np.load(f"{dir0}/T_map2end_{i}.npy")
    B.append(T_map2end)

print(A)
print(B)

# A, Bをリストから回転と並進に分ける
A_rotations = [cv2.Rodrigues(T[:3, :3])[0] for T in A]
A_translations = [T[:3, 3] for T in A]

B_rotations = [cv2.Rodrigues(T[:3, :3])[0] for T in B]
B_translations = [T[:3, 3] for T in B]

# Tsai-Lenz法を使用してハンド・アイキャリブレーションを行う
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base=A_rotations,
    t_gripper2base=A_translations,
    R_target2cam=B_rotations,
    t_target2cam=B_translations,
    method=cv2.CALIB_HAND_EYE_TSAI
)

# 結果の表示
print("回転行列 (R):\n", R_cam2gripper)
print("並進ベクトル (t):\n", t_cam2gripper)

