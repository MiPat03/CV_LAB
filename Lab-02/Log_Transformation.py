import cv2
import numpy as np

img = cv2.imread('test.jpg')

# Ensure all pixel values are positive by adding a small constant
epsilon = 1e-8  # Small constant to prevent division by zero
img_positive = img + epsilon

c = 255 / (np.log(1 + np.max(img_positive)))
log_transformed = c * (np.log(img_positive + 1))
log_transformed = np.array(log_transformed, dtype=np.uint8)

cv2.imwrite('log_transformed.jpg', log_transformed)
cv2.imshow('log_transform', log_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
