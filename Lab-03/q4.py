import cv2
import matplotlib.pyplot as plt

def detect_edges(image_path, low_threshold, high_threshold):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(original_image, low_threshold, high_threshold)

    return edges

input_image_path = 'reference.jpeg'

low_threshold = 50
high_threshold = 150

edge_image = detect_edges(input_image_path, low_threshold, high_threshold)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap='gray')
plt.axis('off')
plt.title('Edge Detected Image')

plt.show()
