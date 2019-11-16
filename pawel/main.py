import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os #read all files in directory
from scipy import ndimage as ndi
import tranformations as t
import time
import HuMoments as hu
import gc

gc.collect()
easy = []
medium = []
hard = []
paths = ["images/easy", "images/medium", "images/hard"]
#paths = ["images/easy"]



#read images from files
def readImages():
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                print(file)
                if(path == paths[0]):
                    easy.append(cv.imread(os.path.join(r, file), 1))
                elif(path == paths[1]):
                    medium.append(cv.imread(os.path.join(r, file), 1))
                elif(path == paths[2]):
                    hard.append(cv.imread(os.path.join(r, file), 1))


#show images
def showImages(images, number, rows, columns):
    plt.figure(figsize=(20, 20))
    for i in range(1, number + 1):
        plt.subplot(rows, columns, i)
        plt.imshow(images[i - 1], cmap = "gray") #bgr
        #plt.imshow(images[i - 1][:, :, ::-1]) #rgb
    plt.show()
    plt.close()
    plt.clf()


def showCropped(cropped, original):
    plt.subplot(121), plt.imshow(original), plt.title('Input')
    plt.subplot(122), plt.imshow(cropped), plt.title('Output')
    plt.show()
    time.sleep(0.5)
    plt.close()
    plt.clf()


#main function
def transformImages(images):

    for i in range(len(images)):
        print("Start of tranformations...")
        grayTemp = t.bgr2gray(images[i])
        contrastTemp = t.contrast(grayTemp, 48.5)
        gammaTemp = t.gamma(contrastTemp, 0.55)
        blurTemp = t.blur(gammaTemp, 15, 15)
        threshTemp = t.adaptiveThresh(blurTemp)
        erosionTemp = t.erosion(threshTemp, 4)
        openingTemp = t.opening(erosionTemp)

        contoursTemp = t.contours(openingTemp) #finding contours
        contoursAll = contoursTemp
        contoursFinal = []

        #delete too small and too big contours and take these with four corners
        contoursTemp = [contour for contour in contoursTemp if(cv.contourArea(contour) > t.MINIMUM_SIZE_CONTOUR and cv.contourArea(contour) < t.MAXIMUM_SIZE_CONTOUR and len(cv.approxPolyDP(contour, 0.025 * cv.arcLength(contour, True), True))
 == 4)]

        #if there are not contours that fit in above conditions - find again
        #if(len(contoursTemp) == 0): contoursTemp = t.contours(openingTemp)

        #create list with areas of contours
        areas = []
        for contour in contoursTemp:
            areas.append(int(cv.contourArea(contour)))

        #if there are no contours - start new iteration
        if(len(areas) == 0): continue

        medianArea = int(np.median(areas))
        tempAreas = sorted(areas)

        #find contour that area fits to median area
        for j in range(len(tempAreas)):
            if(tempAreas[j] > medianArea):
                medianArea = tempAreas[j - 1]
                break

        #find median contour in original list
        medianAreaIndex = areas.index(medianArea)
        medianContour = contoursTemp[medianAreaIndex]


        # take contours that fit to median contour and their area is bigger than half of median area
        for contour in contoursAll:
            if(cv.matchShapes(contour, medianContour, 1, 0.0) < 0.3 and cv.contourArea(contour) > cv.contourArea(medianContour)*0.5):
                contoursFinal.append(contour)


        #crop contours from original image
        croppedContours = []
        for contour in contoursFinal:
            croppedContours.append([t.crop(contour)])


        detectedContours = []

        #loop through all contours and create new image with them
        for points in croppedContours:
            pts1 = np.float32([[points[0][3]], [points[0][2]], [points[0][0]], [points[0][1]]])
            pts2 = np.float32([[0, 0], [points[0][-1][0], 0], [0, points[0][-1][1]], [points[0][-1][0], points[0][-1][1]]])
            M = cv.getPerspectiveTransform(pts1, pts2)
            croppedImage = cv.warpPerspective(images[i], M, (int(points[0][-1][0]), int(points[0][-1][1])))

            #uncomment this line if you want to see cropped and original image
            #showCropped(croppedImage, images[i])
            print("Before hu.readCard")
            cardAttributes = hu.readCard(croppedImage)
            print(cardAttributes)
            detectedContours.append([cardAttributes[0] + " " + cardAttributes[1] + " " + cardAttributes[2] + " " + str(cardAttributes[3]), points[0][3]])

            #TODO implement detection function
            #detectedContours.append(result from detection function)

        cv.drawContours(images[i], contoursFinal, -1, (0, 255, 0), 20)

        #printing detected card symbols
        for k in range(len(detectedContours)):
            cv.putText(images[i], detectedContours[k][0], (detectedContours[k][1][0], detectedContours[k][1][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 5)
        #cv.imshow("image", images[i])
        cv.imwrite("out"+str(i)+".jpg",images[i])
        #cv.waitKey(0)
        print("Tranforming photo ended...", i)
        del grayTemp
        del gammaTemp
        del contrastTemp
        del erosionTemp
        del openingTemp
        del blurTemp
        del threshTemp
    del images



readImages()

#transformImages(easy)
#transformImages(medium)
transformImages(hard)

print("Showing images...")
showImages(easy, 4, 2, 2)
#showImages(medium, 15, 5, 3)
#showImages(hard, 16, 4, 4)


del easy
del medium
del hard
cv.destroyAllWindows()