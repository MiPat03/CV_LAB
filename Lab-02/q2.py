import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input and reference images
input_img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
reference_img = cv2.imread('reference.jpeg', cv2.IMREAD_GRAYSCALE)

# Calculate histograms for the input and reference images
hist_input, _ = np.histogram(input_img.flatten(), bins=256, range=[0, 256])
hist_reference, _ = np.histogram(reference_img.flatten(), bins=256, range=[0, 256])

# Calculate cumulative distribution functions (CDFs) for the histograms
cdf_input = hist_input.cumsum()
cdf_reference = hist_reference.cumsum()

# Normalize the CDFs to have values in the range [0, 255]
cdf_input_normalized = (cdf_input - cdf_input.min()) * 255 / (cdf_input.max() - cdf_input.min())
cdf_reference_normalized = (cdf_reference - cdf_reference.min()) * 255 / (cdf_reference.max() - cdf_reference.min())

# Create a lookup table for histogram specification
lut = np.interp(cdf_input_normalized, cdf_reference_normalized, np.arange(256))

# Apply the lookup table to the input image to perform histogram specification
output_img = cv2.LUT(input_img, lut.astype(np.uint8))

# Display the input, reference, and output images
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.title('Input Image')
plt.imshow(input_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Output Image')
plt.imshow(output_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Histogram of Input Image')
plt.hist(input_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.subplot(2, 3, 5)
plt.title('Histogram of Reference Image')
plt.hist(reference_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.subplot(2, 3, 6)
plt.title('Histogram of Output Image')
plt.hist(output_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()