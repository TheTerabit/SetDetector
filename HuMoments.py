import cv2
import imutils
from resizeimage import resizeimage
from skimage import color
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import erosion, dilation

###Image helper functions
def cutOut(image):
    start = 0
    end = len(image) - 1
    begin = True
    center = image[len(image) // 2]
    for i in range(len(center)):
        if center[i] > 0 and begin:
            start = i - 20
            begin = False
        if center[i] == 0 and not begin:
            end = i + 20
            break

    return image[5:-5, start:end]

####Image attributes finding

def pickNumber(image):
    ones = 0
    twos = 0
    threes = 0
    for line in range(0,len(image),10):

        #center = image[len(image) // 2]
        n = 0
        for i in range(1,len(image[line])-1):
            if image[line][i] > 0 and image[line][i+1] == 0:
                n+=1
        if n == 1:
            ones += 1
        elif n == 2:
            twos += 1
        elif n == 3:
            threes += 1


    if ones > twos and ones > threes:
        return 1
    elif twos > ones and twos > threes:
        return 2
    else:
        return 3
    #return n

def pickColor(image, thresh):
    color = []
    for i in range(len(thresh)):
        for j in range(len(thresh[i])):
            if thresh[i][j] > 0:
                color.append(image[i][j])
                #print(image[i][j])
    r = 0
    g = 0
    b = 0
    rm = []
    gm = []
    bm = []
    for i in color:
        #r += i[0]
        #g += i[1]
        #b += i[2]
        rm.append(i[2])
        gm.append(i[1])
        bm.append(i[0])
    #print(rm)
    rm.sort()
    gm.sort()
    bm.sort()
    #print(rm)
    rm = rm[len(rm) // 2]
    gm = gm[len(gm) // 2]
    bm = bm[len(bm) // 2]
    #r /= len(color)
    #g /= len(color)
    #b /= len(color)

    if bm > rm and bm > gm:
        return 'violet'
    elif rm > bm and rm > gm:
        return 'red'
    else:
        return 'green'

    return (r, g, b, rm, gm, bm)
    #return ('red', 'green', 'violet')

def pickFilling(image, imageFilled):
    #print(image)
    before = 0
    after = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0:
                before += 1
            if imageFilled[i][j] > 0:
                after += 1

    filled = before / after
    print(filled)
    if filled > 0.9:
        return 'full'
    elif filled > 0.3:
        return 'striped'
    else:
        return 'empty'
    #return ('full', 'empty', 'striped')

def pickShape(thresh):#picking the specyfic shape - only your shape as input
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    print(huMoments)
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
    #print((d,w,o))
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

def checkPixels(image):
    s=0
    for i in image:
        s += sum(i) // 255
    if s > 100:
        return False
    else:
        return True
def prepareImage(image):
    im = color.rgb2grey(image)
    #im = cv2.blur(im, (5, 5))
    im = contrast(im, 1)
    #im = resizeimage.resize_cover(im, [len(im[0])*2, len(im)*2], validate=False)
    im = gamma(im, 0.25)

    im = tresh(0.7, im)
    #t =0.5
    #i = im
    #while(checkPixels(i)):
    #    i = tresh(t, im)
    #    t += 0.1

    #im = i
    #im = cv2.fastNlMeansDenoising(im, 300)
    #im = cv2.blur(im, (100, 100))
    im = dilation(im)
    im = erosion(im)
    #print(im[len(im)//2])
    #im = cv2.medianBlur(im, 3)
    #im = imfill(im, 'holes');
    #for i in im:
    #    print(i)

    #print(im[len(im)//2])
    return im
def prepareImageForColoring(image):
    im = color.rgb2grey(image)
    #im = contrast(im, 1.0)
    # im = gamma(im, 0.5)
    t =0.2
    i = im
    i = tresh(t, im)
    while(checkPixels(i)):
        i = tresh(t, im)
        t += 0.1
    #print (t - 0.1)
    im = i
    #im = cv2.fastNlMeansDenoising(im, 300)
    #im = cv2.blur(im, (100, 100))
    im = dilation(im)
    im = erosion(im)
    #print(im[len(im)//2])
    #im = cv2.medianBlur(im, 3)
    #im = imfill(im, 'holes');
    #for i in im:
    #    print(i)

    #print(im[len(im)//2])
    return im

####main
def readCard(imageName):
    image = cv2.imread(imageName)

    #im = color.rgb2grey(image)
    thresh = prepareImage(image)
    forColoring = prepareImageForColoring(image)
    imageFilled = fillIn(thresh)
    cutImage = cutOut(imageFilled)
    number = pickNumber(imageFilled)
    color = pickColor(image, forColoring)
    filling = pickFilling(thresh, imageFilled)
    #thresh = thresh[5:-5, :-185]#diamond
    shape = pickShape(cutImage)
    #imageFilled = imageFilled[5:-5, 50:]#oval
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
    # show the output image
    print((shape, color, filling, number))
    #cv2.imshow("Image", image)
    cv2.imshow("Image", thresh)
    cv2.waitKey(0)

    return (shape, color , filling, number)


#color ok
#assert readCard("k13.jpg") == ('oval', 'violet', 'striped', 3), "erorr"
#assert readCard("k14.jpg") == ('wave', 'violet', 'empty', 1), "erorr"
#assert readCard("k15.jpg") == ('oval', 'red', 'striped', 1), "erorr"
#assert readCard("k16.jpg") == ('diamond', 'green', 'empty', 1), "erorr" # dziwne rzeczy w rogu - daje striped :(
#assert readCard("k17.jpg") == ('oval', 'violet', 'full', 1), "erorr" # diamond -> oval
#assert readCard("k18.jpg") == ('diamond', 'red', 'empty', 1), "erorr" # wave -> diamond
#assert readCard("k19.jpg") == ('diamond', 'green', 'full', 3), "erorr" #trudne oval -> diamond
#assert readCard("k20.jpg") == ('wave', 'red', 'empty', 2), "erorr" #ok
#assert readCard("k21.jpg") == ('wave', 'green', 'striped', 2), "erorr" # empty -> striped
#assert readCard("k22.jpg") == ('wave', 'violet', 'empty', 3), "erorr"#ok
#assert readCard("k23.jpg") == ('diamond', 'red', 'empty', 3), "erorr" #ok
#assert readCard("k24.jpg") == ('wave', 'red', 'striped', 1), "erorr"# empty -> striped

c = ["k1.png","k2.png","k3.png","k4.png","k5.png","k6.png","k7.png","k8.png","k9.png","k10.png","k11.png","k12.png",]
#assert readCard(c[10]) == ('oval', 'green', 'empty', 2), "erorr" #ok
#assert readCard(c[0]) == ('wave', 'violet', 'full', 1), "erorr" #ok
#assert readCard(c[1]) == ('wave', 'green', 'striped', 3), "erorr" #ok
#assert readCard(c[2]) == ('diamond', 'violet', 'empty', 2), "erorr" #ok
#assert readCard(c[3]) == ('oval', 'red', 'full', 1), "erorr" #ok
#assert readCard(c[4]) == ('wave', 'red', 'striped', 2), "erorr" #ok
assert readCard(c[5]) == ('diamond', 'red', 'full', 1), "erorr" #ok


#for i in c:
#    readCard(i)