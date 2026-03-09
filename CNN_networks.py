import os
import cv2
from matplotlib import pyplot as plt

image_path = r"F:\StudyatCLass\Study\class\DeepLearningandUngdung\image1.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise RuntimeError("Không thể đọc ảnh (cv2.imread trả về None).")

gray_1 = image[::10, ::10]

height, width = image.shape

new_width = width // 10
new_height = height // 10
small_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

cv2.imshow("Original Image", image)
cv2.imshow("Downsampled Image", gray_1)
cv2.imshow("Resized Image", small_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.title("Original")
# plt.imshow(image, cmap="gray")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Downsampled (every 10th pixel)")
# plt.imshow(gray_1, cmap="gray")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("Resize 1/10)")
# plt.imshow(small_image, cmap="gray")
# plt.axis("off")

# plt.show()
