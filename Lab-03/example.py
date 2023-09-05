import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('reference.jpeg')

if image is None:
    print('Could not read image')

#kernel1 -> Identity Filter
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

kernel2 = np.ones((5, 5), np.float32) / 25
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

# Create a single window for displaying images
plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(131)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Display the identity filter result
plt.subplot(132)
plt.imshow(cv2.cvtColor(identity, cv2.COLOR_BGR2RGB))
plt.title('Identity Filter')
plt.axis('off')

# Display the kernel-blurred result
plt.subplot(133)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Kernel Blur')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the identity filter result
cv2.imwrite('identity.jpg', identity)

# Save the kernel-blurred result
cv2.imwrite('Blur_Kernel.jpg', img)
