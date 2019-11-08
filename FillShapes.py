import cv2
import numpy as np

# Read image
im_in = cv2.imread("k3.png", cv2.IMREAD_GRAYSCALE)

# Threshold
th, im_th = cv2.threshold(im_in, 127, 255, cv2.THRESH_BINARY)

im_th = 255 - im_th
# Copy the thresholded image
im_floodfill = im_th.copy()

# Mask used to flood filling.
# NOTE: the size needs to be 2 pixels bigger on each side than the input image
h, w = im_th.shape[:2]
print(h, w)
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (20,20), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground
im_out = im_th | im_floodfill_inv

im_out=im_out[10:-10, 190:-60]
moments = cv2.moments(im_out)

# Calculate Hu Moments
huMoments = cv2.HuMoments(moments)
print(huMoments)

# Display images.
cv2.imwrite("diamond2.png", im_out)
cv2.imshow("Image", im_out)

cv2.waitKey(0)