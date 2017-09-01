import cv2
from spatialAppearanceFeature import crop_center

img = cv2.imread("George_W_Bush_0179.jpg")
img = crop_center(img,150,150)

cv2.imwrite("center179.png", img)  
