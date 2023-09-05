import cv2
import matplotlib.pyplot as plt

# Load an image from the specified file path in grayscale mode
img = cv2.imread('lowContrast.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization using cv2.equalizeHist()
equalized_img = cv2.equalizeHist(img)

# Calculate histograms for both original and equalized images
hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Histogram of Original Image')
plt.hist(img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.subplot(2, 2, 4)
plt.title('Histogram of Equalized Image')
plt.hist(equalized_img.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()