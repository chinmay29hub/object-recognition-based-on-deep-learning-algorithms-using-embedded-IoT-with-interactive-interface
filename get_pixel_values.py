import cv2

img = cv2.imread('3.jpg')
x = 100
y = 200
pixel_value = img[x, y]

print("Pixel value at ({}, {}) is {}".format(x, y, pixel_value))
