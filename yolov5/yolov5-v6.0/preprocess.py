# 彩色图像均衡化
import cv2
import numpy as np


def equalize_Hist_color(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image


def equalize_clahe_color_hsv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image


def equalize_clahe_color_lab(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eq_l = cla.apply(l)
    eq_image = cv2.cvtColor(cv2.merge([eq_l, a, b]), cv2.COLOR_Lab2BGR)
    return eq_image


def equalize_clahe_color_yuv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_Y = cla.apply(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y, U, V]), cv2.COLOR_YUV2BGR)
    return eq_image


def equalize_clahe_color(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image


image = cv2.imread("test.jpg", 1)
# 彩色图像应用  HE和CLAHE
image_he_color = equalize_Hist_color(image)
image_clahe_color = equalize_clahe_color(image)
image_clahe_color_lab = equalize_clahe_color_lab(image)
image_clahe_color_hsv = equalize_clahe_color_hsv(image)
image_clahe_color_yuv = equalize_clahe_color_yuv(image)

cv2.imshow("raw_image", image)
cv2.imwrite("./pre-images/raw_image.jpg", image)

# cv2.imshow("image_he_color", image_he_color)
cv2.imwrite("./pre-images/he-rgb.jpg", image_he_color)

# cv2.imshow("image_clahe_color", image_clahe_color)
cv2.imwrite("./pre-images/clahe-rgb.jpg", image_clahe_color)

# cv2.imshow("image_clahe_color_lab", image_clahe_color_lab)
cv2.imwrite("./pre-images/clahe-lab.jpg", image_clahe_color_lab)

# cv2.imshow("image_clahe_color_hsv", image_clahe_color_hsv)
cv2.imwrite("./pre-images/clahe-hsv.jpg", image_clahe_color_hsv)

# cv2.imshow("image_clahe_color_yuv", image_clahe_color_yuv)
cv2.imwrite("./pre-images/clahe-yuv.jpg", image_clahe_color_yuv)\

cv2.waitKey(0)


