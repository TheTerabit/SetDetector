import cv2
import imutils
from skimage import color
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import erosion, dilation


####Image attributes finding
def pickNumber(image):
    center = image[len(image) // 2]
    n = 0
    for i in range(1,len(center)-1):
        if center[i] > 0 and center[i+1] == 0:
            n+=1
    return n

def pickColor(image):
    return ('red', 'green', 'violet')

def pickFilling(image, imageFilled):
    print(image)
    before = 0
    after = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0:
                before += 1
            if imageFilled[i][j] > 0:
                after += 1

    filled = before / after
    if filled > 0.9:
        return 'full'
    elif filled > 0.5:
        return 'striped'
    else:
        return 'empty'
    #return ('full', 'empty', 'striped')

def pickShape(thresh):#picking the specyfic shape - only your shape as input
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    #print(huMoments)
    diamond = [[ 8.17072052e-04],
    [ 2.32278146e-07],
    [ 1.02494107e-12],
    [ 1.59064485e-13],
    [-4.40488960e-27],
    [-7.66380864e-17],
    [-6.40745333e-26]]
    wave = [[9.95799976e-04],
     [5.18754061e-07],
     [6.95797209e-12],
     [2.36847316e-12],
     [9.25072564e-24],
     [1.61455840e-15],
     [2.62108125e-24]]
    oval = [[7.71516107e-04],
     [1.97717475e-07],
     [6.67240715e-14],
     [5.78949686e-15],
     [6.77168851e-29],
     [1.26244655e-18],
     [9.14466650e-29]]

    d = 0
    w = 0
    o = 0
    for i in range(len(huMoments)):
        d = d + abs(huMoments[i][0] - diamond[i])
        w = w + abs(huMoments[i][0] - wave[i])
        o = o + abs(huMoments[i][0] - oval[i])

    if d < w and d < o:
        return 'diamond'
    elif w < d and w < o:
        return 'wave'
    else:
        return 'oval'

def fillIn(im_th):
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # NOTE: the size needs to be 2 pixels bigger on each side than the input image
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (20, 20), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground
    im_out = im_th | im_floodfill_inv
    return im_out

#####Image preparation
def tresh(t, x):
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

def prepareImage(image):
    im = color.rgb2grey(image)
    #im = contrast(im, 1.0)
   # im = gamma(im, 0.5)
    im = tresh(0.5, im)
    im = dilation(im)
    #im = erosion(im)
    #print(im[len(im)//2])
    #im = cv2.medianBlur(im, 3)
    #im = imfill(im, 'holes');
    #for i in im:
    #    print(i)

    #print(im[len(im)//2])
    return im


image = cv2.imread("k3.png")

im = color.rgb2grey(image)
thresh = prepareImage(image)
imageFilled = fillIn(thresh)
print(pickNumber(imageFilled))
print(pickColor(thresh))
print(pickFilling(thresh, imageFilled))
#thresh = thresh[5:-5, :-185]#diamond
imageFilled = imageFilled[5:-5, 50:]#oval
#thresh = thresh[5:-5, 50:-250]#wave
#--
#cnts = find_contours(thresh, 0.7)
#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
# loop over the contours
#for c in cnts:

# compute the center of the contour, then detect the name of the
# shape using only the contour


# Calculate Hu Moments


print(pickShape(imageFilled))

# show the output image
cv2.imshow("Image", imageFilled)

cv2.waitKey(0)