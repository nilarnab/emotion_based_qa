import cv2
import numpy as np

img = cv2.imread("./winCeleb.jpg")

resize_img = cv2.resize(img, (100, 100))

print("new image shape ", np.shape(resize_img))

cv2.imwrite("./reshape_img.jpg", resize_img)