import cv2

img = cv2.imread('test.jpg')
cv2.imshow('test', img)
cv2.waitKey(1000)
print('Shape of image: ',img.shape)
b, g, r = img[100, 100]
print('RGB Values')
print("R: ",r)
print("G: ",g)
print("B: ",b)
