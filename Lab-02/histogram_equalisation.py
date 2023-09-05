import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Separate the image into RGB channels
    r, g, b = cv2.split(image)

    # Equalize each channel separately
    r_equalized = equalize_channel(r)
    g_equalized = equalize_channel(g)
    b_equalized = equalize_channel(b)

    # Merge the equalized channels back into a colored image
    equalized_image = cv2.merge((r_equalized, g_equalized, b_equalized))

    return equalized_image

def equalize_channel(channel):
    # Calculate the histogram of the channel
    histogram = np.zeros(256, dtype=int)
    for row in channel:
        for pixel_value in row:
            histogram[pixel_value] += 1

    # Calculate the cumulative histogram
    cumulative_histogram = np.cumsum(histogram)

    # Calculate the mapping function
    mapping_function = (cumulative_histogram - cumulative_histogram.min()) * 255 / (cumulative_histogram.max() - cumulative_histogram.min())

    # Apply the mapping function to equalize the channel
    equalized_channel = np.zeros_like(channel)
    for i in range(256):
        equalized_channel[channel == i] = mapping_function[i]

    return equalized_channel

if __name__ == "__main__":
    # Load the input colored image
    input_image = cv2.imread("lowContrast.jpeg")

    # Perform histogram equalization
    equalized_image = histogram_equalization(input_image)

    # Create subplots for displaying images and histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # Display the original colored image
    axs[0, 0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image")

    # Display the equalized colored image
    axs[0, 1].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Equalized Image")

    # Calculate and plot the histogram of the original image (for one channel)
    axs[1, 0].hist(input_image[:,:,0].ravel(), bins=256, range=(0, 256), color='b', alpha=0.7, label='Red')
    axs[1, 0].hist(input_image[:,:,1].ravel(), bins=256, range=(0, 256), color='g', alpha=0.7, label='Green')
    axs[1, 0].hist(input_image[:,:,2].ravel(), bins=256, range=(0, 256), color='r', alpha=0.7, label='Blue')
    axs[1, 0].set_title("Original Histogram")

    # Calculate and plot the histogram of the equalized image (for one channel)
    axs[1, 1].hist(equalized_image[:,:,0].ravel(), bins=256, range=(0, 256), color='b', alpha=0.7, label='Red')
    axs[1, 1].hist(equalized_image[:,:,1].ravel(), bins=256, range=(0, 256), color='g', alpha=0.7, label='Green')
    axs[1, 1].hist(equalized_image[:,:,2].ravel(), bins=256, range=(0, 256), color='r', alpha=0.7, label='Blue')
    axs[1, 1].set_title("Equalized Histogram")

    # Adjust layout to fit the smaller size
    plt.tight_layout(pad=1.0)

    # Show the subplots
    plt.show()
