import cv2
import numpy as np

img = cv2.imread('test.jpg')
cv2.imshow('Original Image', img)

down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)

up_width = 600
up_height = 400
up_points = (up_width, up_height)
resized_up = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Resized Down',resized_down)
cv2.waitKey(0)
cv2.imshow('Resized Up',resized_up)
cv2.waitKey(0)

cv2.destroyAllWindows()