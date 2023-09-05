import cv2
import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
p0 = 100, 10
p1 = 200, 90

img = np.zeros((500, 500, 3), np.uint8)
cv.rectangle(img, p0, p1, BLUE, 2)
cv2.imshow('RGB', img)
cv.waitKey(0)
cv2.destroyAllWindows()
