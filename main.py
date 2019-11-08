import cv2
from pylab import *
import matplotlib.pyplot as plt
from skimage import measure
from skimage import io
import skimage.morphology as mp
from matplotlib import pylab as plt
import numpy as np
from skimage import color

#from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = 10 #cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        print(approx);
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def tresh(t, x):
        #warnings.simplefilter("ignore")
        binary = (x > t) * 1.0
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
        image = color.rgb2grey(image)
        im = contrast(image, 1.0)
        im = gamma(im, 0.8)
        im = tresh(0.7, im)
        im = mp.erosion(im)
        return im

def downloadImages():
        url = "http://www.cs.put.poznan.pl/mtomczyk/kck/Lab4_images/planes/"
        #images = ["samolot05.jpg", "samolot07.jpg", "samolot08.jpg", "samolot09.jpg", "samolot10.jpg", "samolot11.jpg"]
        #sets = ["L1.jpg", "S1.jpg", "S2.jpg", "S3.jpg", "T1.jpg"]
        cards = ["k1.png", "k2.png", "k3.png", "k4.png"]
        planes = []
        for i in cards:
            # planes.append(io.imread(url + i))
            planes.append(io.imread(i))
        return planes

def drawPlanes(planes):

        sd = ShapeDetector()


        plt.figure(figsize=(30, 20))
        for i, plane in enumerate(planes):
            plane = prepareImage(plane)
            contours = measure.find_contours(plane, 0.7)
            resized = imutils.resize(plane, width=300)
            ratio = plane.shape[0] / float(resized.shape[0])
            for c in contours:
                # compute the center of the contour, then detect the name of the
                # shape using only the contour
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape = sd.detect(c)

                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(plane, [c], -1, (0, 255, 0), 2)
                cv2.putText(plane, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

                # show the output image
                cv2.imshow("Image", plane)
                cv2.waitKey(0)
            plane = tresh(plane, 0)
            fig, ax = plt.subplots()
            ax.imshow(1 - plane, cmap=plt.cm.gray)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], '-w')
            plt.show()


planes = downloadImages()
drawPlanes(planes)



