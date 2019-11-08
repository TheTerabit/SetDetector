import cv2
import imutils
from skimage import color
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import erosion, dilation


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

def fillIn(image):
    im = image.copy()

    for i in range(20,len(image)-20):
        fill = False
        for j in range(20,len(image[i])-20):
            if image[i][j] == 255 and image[i][j-1] == 0:
                fill = not fill
            if fill:
                im[i][j] = 255;

    return im

def prepareImage(image):
    im = color.rgb2grey(image)
    #im = contrast(im, 1.0)
   # im = gamma(im, 0.5)
    im = tresh(0.5, im)
    im = dilation(im)
    #im = erosion(im)
    print(im[len(im)//2])
    #im = cv2.medianBlur(im, 3)

    for i in im:
        print(i)
    im = fillIn(im)
    print(im[len(im)//2])
    return im


image = cv2.imread("k3.png")

#resized = imutils.resize(image, width=100)
#ratio = image.shape[0] / float(resized.shape[0])

im = color.rgb2grey(image)
thresh = prepareImage(image)


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
# show the output image
cv2.imshow("Image", thresh)

cv2.waitKey(0)