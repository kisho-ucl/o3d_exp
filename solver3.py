import numpy as np
from scipy.optimize import least_squares

# 変換行列を読み込む
n = 15
dir0 = '/Users/kisho/open3d/o3d_exp/datas'

# カメラからマーカーへの変換行列 A_i
T_cam2marker = []
for i in range(n):
    T = np.load(f"{dir0}/T_cam2marker_{i}.npy")
    T_cam2marker.append(T)

# マップかエンドエフェクタへの変換行列 B_i
T_map2end = []
for i in range(n):
    T = np.load(f"{dir0}/T_map2end_{i}.npy")
    T_map2end.append(T)

As = []
for i in range(1,n):
    A = T_cam2marker[i-1]@np.linalg.inv(T_cam2marker[i])
    As.append(A)

Bs = []
for i in range(1,n):
    B = np.linalg.inv(T_map2end[i-1])@T_map2end[i-1]
    Bs.append(B)


# 最適化問題の定義
def hand_eye_residuals(params, A_list, B_list):
    # パラメータを4x4行列に変換
    X = params.reshape(4, 4)

    residuals = []
    for A, B in zip(A_list, B_list):
        A_rotation = A[:3, :3]
        A_translation = A[:3, 3]
        B_rotation = B[:3, :3]
        B_translation = B[:3, 3]

        # B' = X * A の計算
        B_prime = X @ A
        B_prime_rotation = B_prime[:3, :3]
        B_prime_translation = B_prime[:3, 3]

        # 残差を計算
        residuals.extend((B_rotation - B_prime_rotation).flatten())
        residuals.extend((B_translation - B_prime_translation).flatten())

    return np.array(residuals)

# 初期推定値
initial_X = np.eye(4).flatten()

# 最適化の実行
result = least_squares(hand_eye_residuals, initial_X, args=(As, Bs))

# 最適化された変換行列
optimized_X = result.x.reshape(4, 4)

print("最適化された変換行列 X:")
print(optimized_X)

np.save("{dir0}/T_cam2end.npy",optimized_X)
