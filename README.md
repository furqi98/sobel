import cv2
import numpy as np
import os

# Input and output directories
input_folder = "Udacity_Dataset/"
output_folder = "Sobel_output/"
os.makedirs(output_folder, exist_ok=True)

# Process all images
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        # Convert gradients to uint8
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        # Combine the Sobel images
        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        # Save the result
        cv2.imwrite(os.path.join(output_folder, filename), sobel_combined)

print("Sobel Edge Detection applied to all images.")
