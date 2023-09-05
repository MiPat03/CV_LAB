import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_specification(input_image, reference_image):
    # Calculate histograms of input and reference images for each channel
    input_hist_b = cv2.calcHist([input_image], [0], None, [256], [0, 256])
    input_hist_g = cv2.calcHist([input_image], [1], None, [256], [0, 256])
    input_hist_r = cv2.calcHist([input_image], [2], None, [256], [0, 256])

    reference_hist_b = cv2.calcHist([reference_image], [0], None, [256], [0, 256])
    reference_hist_g = cv2.calcHist([reference_image], [1], None, [256], [0, 256])
    reference_hist_r = cv2.calcHist([reference_image], [2], None, [256], [0, 256])

    # Create mapping functions for each channel
    mapping_function_b = create_mapping_function(input_hist_b, reference_hist_b)
    mapping_function_g = create_mapping_function(input_hist_g, reference_hist_g)
    mapping_function_r = create_mapping_function(input_hist_r, reference_hist_r)

    # Apply the mapping functions to each channel
    matched_image = cv2.LUT(input_image, mapping_function_b)
    matched_image[:,:,1] = cv2.LUT(input_image[:,:,1], mapping_function_g)
    matched_image[:,:,2] = cv2.LUT(input_image[:,:,2], mapping_function_r)

    return matched_image

def create_mapping_function(input_hist, reference_hist):
    # Calculate cumulative histograms
    input_cumulative = np.cumsum(input_hist)
    reference_cumulative = np.cumsum(reference_hist)

    # Normalize cumulative histograms
    input_cumulative_normalized = (input_cumulative - input_cumulative.min()) * 255 / (input_cumulative.max() - input_cumulative.min())
    reference_cumulative_normalized = (reference_cumulative - reference_cumulative.min()) * 255 / (reference_cumulative.max() - reference_cumulative.min())

    # Create a mapping function
    mapping_function = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapping_function[i] = np.argmin(np.abs(input_cumulative_normalized - reference_cumulative_normalized[i]))

    return mapping_function

if __name__ == "__main__":
    # Load the input colored image and reference image
    input_image = cv2.imread("lowContrast1.jpg")
    reference_image = cv2.imread("reference.jpeg")

    # Perform histogram specification
    matched_image = histogram_specification(input_image, reference_image)

    # Display the input image, reference image, and the matched image
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title("Matched Image")

    plt.tight_layout()
    plt.show()
