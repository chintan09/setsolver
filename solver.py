#!/usr/bin/env python

import numpy as np
import cv2
import sys

## TODO: filter the image & run through canny edge detector

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def show_image(img, resize=True):
    if resize:
        w = int(img.shape[0] * 0.5)
        h = int(img.shape[1] * 0.5)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA )
    cv2.imshow('image',img)
    k = cv2.waitKey(0) & 0xff
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()


def rectify(h):

    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def warp_transform(contour, img_rect):
    peri = cv2.arcLength(contour,True)
    print peri

    print cv2.isContourConvex(contour)
    approx = cv2.approxPolyDP(contour,0.02*peri,True)
    print "approx", len(approx)
    #show_image(approx, False)

    approx = cv2.approxPolyDP(contour,peri,True)
    print "approx2", len(approx)
    #show_image(approx, False)

    cv2.drawContours(img_rect, [contour], -1, (0, 0, 255), 2)

    #show_image(img_rect, False)
    #transform = cv2.getPerspectiveTransform(approx,img_rect)
    #warp = cv2.warpPerspective(img,transform,(450,450))

    #show_image(warp)

## give an input, detect it's four properties
def detect_card_properties(img, org_img):

    #show_image(img)

    img_copy = img.copy()
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    card_size = img.shape[0] * img.shape[1]
    num_feature = 8
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:num_feature]

    for c in contours:
        contour_ratio = cv2.contourArea(c) / card_size
        if contour_ratio > 0.6 or contour_ratio < 0.01:
            print "Invalid contour size for the card, skip", 100 *(contour_ratio)
            #cv2.drawContours(org_img, [c], -1, (0, 0, 255), 2)
            continue

        print "Contour percentage: ", 100 *(contour_ratio)

        cv2.drawContours(org_img, [c], -1, (0, 255, 0), 2)

        '''
        hu = cv2.HuMoments(cv2.moments(c)).flatten()

        if hu[0] < 0.207:
            print "OVAL:", hu[0]
        elif hu[0] > 0.23:
            print "squiggle:", hu[0]
        else:
            print "diamond:", hu[0]
        '''
    #show_image(org_img)



def detect_contours(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #laplacian = cv2.Laplacian(gray,cv2.CV_8FC1)
    #show_image(laplacian)

    test = "binary"
    if test == "binary":

        blur = cv2.medianBlur(gray,5)
       #show_image(blur, True)
        '''
        res, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        res, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        res, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        THRESH_TOZERO
        '''
        res, filtered_image = cv2.threshold(blur, 120, 255, cv2.THRESH_TOZERO)

    elif test == "otsu":
        ret3,filtered_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        filtered_image_ = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        filtered_image = cv2.medianBlur(filtered_image_,5)

    filtered_image_copy = filtered_image.copy()

    contours, hierarchy = cv2.findContours(filtered_image_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #show_image(filtered_image)

    image_size = filtered_image.shape[0] * filtered_image.shape[1]
    numcards = 20

    tup = sorted(zip(contours, hierarchy[0]), key= lambda c:cv2.contourArea(c[0]), reverse=True)[:numcards]
    for t in tup:
        print t[1], cv2.contourArea(t[0]), type(t[0]), type(contours), type(t[1])

    contours2 = sorted(contours, key=cv2.contourArea, reverse=True)[:numcards]

    for card in contours2:
        print cv2.contourArea(card)
        card_percentage = 100 * (cv2.contourArea(card) / image_size)
        if card_percentage > 50:
            print "Contour too big, Skipping! ", card_percentage
            continue

        x, y, w, h = cv2.boundingRect(card)

        # !! fit a rectangle & print on the screen
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 2)

        ## print this to a file or so
        filt_img_rect = filtered_image[y:y+h, x:x+w]
        img_rect = img[y:y+h, x:x+w]

        detect_card_properties(filt_img_rect, img_rect)

        ## Draw contours on the image
        cv2.drawContours(img, [card], -1, (0, 255, 0), 2)
    show_image(img)

def main():
    img = cv2.imread(sys.argv[1])
    detect_contours(img)



if __name__ == "__main__":
    main()
