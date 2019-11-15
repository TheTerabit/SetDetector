# import the necessary packages
import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.005 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        print(len(approx))
        if len(approx) < 11:
            shape = "diamond"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) < 18:
           shape = "oval"

        else:
            shape = "wave"

        # return the name of the shape
        return shape


# import the necessary packages
#from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
def getShape(image):
    #image = cv2.imread(image)
    #resized = imutils.resize(image, width=300)
    #ratio = image.shape[0] / float(resized.shape[0])
    image = image[1:-1 , 1:-1]
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    #gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = image
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        #cX = int((M["m10"] / M["m00"]) * ratio)
        #cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        return shape
        print("new:" + shape)
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)