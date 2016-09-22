#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Testing orb
refs:
    Documentation
    http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html
    Example
    http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
    Tutorial
    http://docs.opencv.org/master/dc/d16/tutorial_akaze_tracking.html#gsc.tab=0
    
Constructor:
    cv2.ORB([nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize]]]]]]]])

Operator:
    cv2.ORB.detect(image[, mask]) → keypoints
    cv2.ORB.compute(image, keypoints[, descriptors]) → keypoints, descriptors
    cv2.ORB.detectAndCompute(image, mask[, descriptors[, useProvidedKeypoints]]) → keypoints, descriptors

"""
import numpy as np
from numpy import *
import cv2
import cv2.cv as cv
import time
#setup video capture:
capture_size=(640,420)  # define webcam-display dimensions
#capture_size=(1280,840)  # define webcam-display dimensions
cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT,capture_size[1])
ret,im = cap.read()

# Parameters for ORB Constructor
nfeatures = 1000
scaleFactor = 1.2
nlevels = 8
edgeThreshold = 31 #default = 31
firstLevel = 0 # Not really optional. "Should be a 0 in the current implmeentation".
WTA_K = 4 # number of points to produce an ORB descriptor element (see documentation for detailed information and other values e.g., 3 and 4)
scoreType = cv2.ORB_FAST_SCORE #default = cv2.ORB_HARRIS_SCORE # HARRIS_SCORE is default. FAST_SOCRE is less stabe but a bit faster to compute
#scoreType = cv2.ORB_HARRIS_SCORE # HARRIS_SCORE is default. FAST_SOCRE is less stabe but a bit faster to compute

# Harris_Score time for 1000keypoints = 0.0131001472473
# Fast_Score time for 1000 keypoints  = 0.00877499580383
patchSize = 31


# initiate the dectector:
# orb = cv2.ORB() #use default values
orb = cv2.ORB(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize) #use user values

frame_count = 0

if ret:
    prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    done = False

    while not done:
    # get grayscale image
        ret,im = cap.read()
        ret = True
        im = cv2.imread('music.png')
        if ret:
            frame_count+=1
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            tic = time.time()
            
            # find the keypoints:
            kp = orb.detect(im,mask=None)
            
            # compute the descriptors:
            kp, des = orb.compute(im, kp)
            
            # Find kepyoints and Compute descriptor in one pass
            # Orb dectection parameters
            #mask = np.ones((im.shape[:2]), dtype = np.uint8) # selecting  mask
            #kp, des = orb.detectAndCompute(im, mask=None)
            toc = time.time()
            
            # draw only keypoint location (not size and orientation)
            im2 = cv2.drawKeypoints(im, kp)#, color=(255,0,0), flags=0)
            
            cv2.imshow("ORB_wtka2.png", im2)
            #cv2.imshow("ORB_wtka3.png", im2)
            #cv2.imshow("ORB_wtka4.png", im2)
            
            cv2.waitKey(0)
            # keystroke:
            #key = cv2.waitKey(1) & 255
            if key ==27 or frame_count == 1: #esc to exit
                done = True
            elif key == ord(' '):#spacebar to save frame
                cv2.imwrite('../data/orb_im.png', im2)
                print 'image saved'

print "Execution time: {}".format(toc-tic)

# one pass = 0.00755596160889 (detectAndCompute): 513 keypoints
#            0.00859093666077 : 1k keypoints
# two pass = 0.00882411003113 (detect then compute): 513 keypoints
#            0.0102207660675  : 1k keypoints
# Short analysis on 1k keypoints
#   Difference = 20msecs favoring one pass
#   Perkeyframe= 20usecs favoring one pass (1M keypoints means single pass is .2secs faster)

des = np.unpackbits(des, axis=1)
print "Descriptor information: ", type(des), des.shape#, des[0,0].dtype, type(des[0,0]) # ndarray
print "One descriptor: ", len(des[0]) , des[0]
print "Keypoint information: ", type(kp), len(kp) # list
i = 100
print "One keypoint {} has reponse={} and scale {} ".format(i, kp[i].response, kp[i].octave)
#while
cv2.destroyAllWindows()
cap.release()