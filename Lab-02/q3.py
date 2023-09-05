import cv2

# Load the image
image_path = 'test.jpg'
image = cv2.imread(image_path)

# Resize the image
new_width = 300
new_height = int(image.shape[0] * (new_width / image.shape[1]))
resized_image = cv2.resize(image, (new_width, new_height))

# Define the coordinates for cropping
x1, y1 = 50, 50  # Top-left corner
x2, y2 = 250, 250  # Bottom-right corner

# Crop the image
cropped_image = image[y1:y2, x1:x2]

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the resized image
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the cropped image
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()