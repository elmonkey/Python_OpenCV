#!/usr/bin/env python
#-*- encoding: utf-8 -*-
'''
Created on Mar 25, 2014
Use opencv to generate and apply a background subtractor to the webcam stream.

opencv method idea obtained from
http://weirdinventionoftheday.blogspot.com/2012/10/opencv-background-subtraction-in-python.html

The code was 99% modified, specially to fit my needs (CT)

@author: Carlos <carlitos408@gmail.com>
'''
import cv2
import cv2.cv as cv
import numpy as np


def applymask(im, mask):
    '''applies a mask to a grayscale image
    1. ensure mask is binary and same # of channels as the image
    '''
    # Apply the mask via bit-wise AND:
    mask  = mask /np.max(mask) # binary [0,1]
    masked = cv2.bitwise_and(im,im,mask = mask)
    return masked
# applymask


# cv2.BackgroundSubtractorMOG([history, nmixtures, backgroundRatio[, noiseSigma]]) 
# → <BackgroundSubtractorMOG object>
bgs = cv2.BackgroundSubtractorMOG(24*60,1,0.9,0.01)

# setup video capture:
capture_size = (640, 480)
cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, capture_size[1])

done = False

# Read the first frame as the background
ret, im = cap.read()
ready = False
ret, im = cap.read()

# Use the very first frame as the background
while not ready:
    if ret:
        fgmask = bgs.apply(im)
        ready = True
    else:
        ret,im = cap.read()
i = 1
while not done:
    ret, im = cap.read()
    fgmask = bgs.apply(im)
    masked = applymask(im, fgmask)
    cv2.imshow('masked feed', masked)
    # keystroke:
    key = cv2.waitKey(5)
    if key ==27: #esc to exit
        done = True
    elif key == ord(' '):#spacebar to save frame
        cv2.imwrite('bgnd_subtracted_screen_shot.jpg', im)
print "fgmask info: ", fgmask.dtype, type(fgmask), np.unique(fgmask)


