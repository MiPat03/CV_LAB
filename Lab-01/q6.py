import cv2

img = cv2.imread('test.jpg')
height, width = img.shape[:2]
center = (width/2, height/2)
rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = 45, scale = 1)
rotated_img = cv2.warpAffine(src = img, M = rotate_matrix, dsize = (width, height))

cv2.imshow('Original Image', img)
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)
# cv2.imwrite('rotated_img.jpg', rotated_img)
