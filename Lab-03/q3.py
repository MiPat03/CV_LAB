import cv2
import numpy as np

def apply_box_filter(image_path, kernel_size):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return

    # Apply a box filter to the image
    box_filtered = cv2.boxFilter(img, -1, (kernel_size, kernel_size))

    return box_filtered

def apply_gaussian_filter(image_path, kernel_size, sigma):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return

    # Apply a Gaussian filter to the image
    gaussian_filtered = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    return gaussian_filtered

def main():
    image_path = "reference.jpeg"  # Replace with the path to your image
    kernel_size = 5
    sigma = 1.0

    # Apply the box filter
    box_filtered = apply_box_filter(image_path, kernel_size)

    # Apply the Gaussian filter
    gaussian_filtered = apply_gaussian_filter(image_path, kernel_size, sigma)

    # Display the original image and both filtered images side by side
    combined = np.hstack((cv2.imread(image_path), box_filtered, gaussian_filtered))
    cv2.imshow("Original vs. Box Filter vs. Gaussian Filter", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
