# -*- coding: utf-8 -*-
"""
ColorSpaces
"""

import numpy as np
import cv2

def apply_operator():
    """
    Apply an gradient of edge operator
    """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return output_im

cap = cv2.VideoCapture(0)

ret,im = cap.read()
if ret:
    prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    done = False

    while not done:
     
        # get a webcam frame
        ret,im = cap.read()
        if ret:
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l = lab[:,:,0]
            yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
            y   = yuv[:,:,0]
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            #canvas = np.hstack((im,np.dstack((gray,gray,gray)),np.dstack((l,l,l)), np.dstack((y,y,y)) ))            
            canvas = np.hstack((im,np.dstack((gray,gray,gray)),lab, yuv ))            
            cv2.imshow('Input|Gray|L|Y', canvas)
            
            #cv2.imshow('Laplacian', laplacian)
            
        
        # poll keystrokes
        key = cv2.waitKey(1) & 255
        # keystroke:
        if key == 27:
            done = True
        elif key == ord(' '):#spacebar to save frame
            cv2.imwrite('../data/colorspaces_im.jpg', canvas)
            print 'image saved' 

#while
cv2.destroyAllWindows()
cap.release()