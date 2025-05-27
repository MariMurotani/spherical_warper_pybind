import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/ebiharamari/Sources/spherical_warper_pybind/build/')
import spherical_warper

img = cv2.imread('window.jpeg')
print(img.shape)

# 入力画像サイズ
H, W = img.shape[:2]

# 正常なequirectangular画像にしたいなら、
# 横: 2 * 焦点距離, 縦: 焦点距離 が理想的
f = 500.0
K = [[f, 0, W / 2],
     [0, f, H / 2],
     [0, 0, 1]]
R = np.eye(3).tolist()  # 無回転


warped = spherical_warper.warp_equirectangular(img, int(W), int(H))

cv2.imwrite("warped.jpg", warped)
