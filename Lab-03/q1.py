import cv2
import numpy as np

def unsharp_masking_colored(image_path, kernel_size=(5, 5), sigma=1.0, alpha=0.1):
    # Read the input image
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Unable to read the image.")
        return

    # Split the image into its color channels
    b, g, r = cv2.split(original_image)

    # Apply Gaussian blur to each color channel
    blurred_r = cv2.GaussianBlur(r, kernel_size, sigma)
    blurred_g = cv2.GaussianBlur(g, kernel_size, sigma)
    blurred_b = cv2.GaussianBlur(b, kernel_size, sigma)

    # Calculate the sharpened channels (Unsharp Mask)
    sharpened_r = r - alpha * (r - blurred_r)
    sharpened_g = g - alpha * (g - blurred_g)
    sharpened_b = b - alpha * (b - blurred_b)

    # Clip the values to ensure they are in the valid range [0, 255]
    sharpened_r = np.clip(sharpened_r, 0, 255).astype(np.uint8)
    sharpened_g = np.clip(sharpened_g, 0, 255).astype(np.uint8)
    sharpened_b = np.clip(sharpened_b, 0, 255).astype(np.uint8)

    # Merge the sharpened channels back into a color image
    sharpened_image = cv2.merge((sharpened_b, sharpened_g, sharpened_r))

    # Display the original image and the sharpened image
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the sharpened image to a file (optional)
    cv2.imwrite('sharpened_image.jpg', sharpened_image)

if __name__ == "__main__":
    image_path = 'reference.jpeg'  # Replace with the path to your input image
    unsharp_masking_colored(image_path)
