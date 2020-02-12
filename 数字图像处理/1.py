from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import numpy as np

# 读取图像
img1 =mpimg.imread("cat.jpg")
plt.imshow(img1)
type(img1)
img1.shape

import cv2
img2 = cv2.imread('cat.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)


from PIL import Image
import matplotlib.pyplot as plt
img3 = Image.open("cat.jpg")
plt.imshow(img3)

# 缩放
ori_img = Image.fromarray(img3)
o_h,o_w = ori_img.size
print()
target_size = (200,200)
new_img = ori_img.resize(target_size)

# 直方图均衡
from PIL import Image
import matplotlib.pyplot as plt
img3 = Image.open("cat.jpg")
img3 = np.array(img3)
plt.hist(img3.ravel())      # 计算直方图
plt.show()
plt.hist(img3.ravel(), bins=255, cumulative=True) # 累计直方图
from PIL import ImageOps
img3_eq = ImageOps.equalize(Image.fromarray(img3))  # 直方图均衡化

# 去噪
import skimage
img3_n = skimage.util.random_noise(img3_eq)  # 加噪声
# 高斯滤波器
from scipy.ndimage.filters import gaussian_filter as gauss
img3_gauss = gauss(img3_n,sigma=1)
# 中值滤波
from scipy.ndimage.filters import median_filter as med
img3_med = med(img3_n,size=3)

# 特征提取
# Sobel边缘检测

