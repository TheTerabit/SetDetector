import cv2
import imutils
from skimage import color
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import erosion, dilation

def pickShape(huMoments):
    return (huMoments - [[ 8.17072052e-04]
 [ 2.32278146e-07]
 [ 1.02494107e-12]
 [ 1.59064485e-13]
 [-4.40488960e-27]
 [-7.66380864e-17]
 [-6.40745333e-26]])



def tresh(t, x):
    # warnings.simplefilter("ignore")
    binary = (x < t) * 255.0
    binary = np.uint8(binary)
    return binary


def contrast(image, perc):
    MIN = np.percentile(image, perc)
    MAX = np.percentile(image, 100 - perc)
    norm = (image - MIN) / (MAX - MIN)
    norm[norm > 1] = 1
    norm[norm < 0] = 0
    return norm


def gamma(image, g):
    return image ** g

def fillIn(im_th):
    # Copy the thresholded image
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # NOTE: the size needs to be 2 pixels bigger on each side than the input image
    h, w = im_th.shape[:2]
    print(h, w)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (20, 20), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground
    im_out = im_th | im_floodfill_inv

    #im_out = im_out[10:-10, 190:-60]
    moments = cv2.moments(im_out)

    return im_out

def prepareImage(image):
    im = color.rgb2grey(image)
    #im = contrast(im, 1.0)
   # im = gamma(im, 0.5)
    im = tresh(0.5, im)
    im = dilation(im)
    #im = erosion(im)
    print(im[len(im)//2])
    #im = cv2.medianBlur(im, 3)
    #im = imfill(im, 'holes');
    #for i in im:
    #    print(i)
    im = fillIn(im)
    print(im[len(im)//2])
    return im


image = cv2.imread("k3.png")

#resized = imutils.resize(image, width=100)
#ratio = image.shape[0] / float(resized.shape[0])

im = color.rgb2grey(image)
thresh = prepareImage(image)

thresh = thresh[5:-5, :-185]
#--
cnts = find_contours(thresh, 0.7)
#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
# loop over the contours
#for c in cnts:

# compute the center of the contour, then detect the name of the
# shape using only the contour
moments = cv2.moments(thresh)

# Calculate Hu Moments
huMoments = cv2.HuMoments(moments)
print(huMoments)
print(pickShape(huMoments))
# show the output image
cv2.imshow("Image", thresh)

cv2.waitKey(0)