import cv2
import numpy as np

# Load the image
image = cv2.imread('reference.jpeg')

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the color range you want to segment (in this example, we'll segment green)
lower_color = np.array([0, 10, 0])  # Lower bound for green in RGB
upper_color = np.array([50, 255, 155])  # Upper bound for green in RGB

# Create a binary mask where the green pixels are within the specified range
mask = cv2.inRange(image_rgb, lower_color, upper_color)

# Apply the mask to the original image to segment the green region
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
