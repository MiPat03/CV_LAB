import cv2
import numpy as np

def calculate_image_gradient(image_path):
    # Read the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Apply Sobel operator for gradient calculation
    gradient_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Normalize the magnitude to the range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display the original image and gradient magnitude
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Gradient Magnitude", gradient_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "reference.jpeg"  # Replace with the path to your image
    calculate_image_gradient(image_path)
