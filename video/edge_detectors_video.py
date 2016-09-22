# -*- coding: utf-8 -*-
"""
Gradients and egdes

"""
import numpy as np
import cv2


cap = cv2.VideoCapture(0)

ret,im = cap.read()
k = 5 # kernel size

if ret:
    prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    done = False

    while not done:
     
        # get a webcam frame
        ret,im = cap.read()
        if ret:
            ybr= cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
            yuv =cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
            l = lab[:,:,0]
            
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            l = gray
            sobelx = cv2.Sobel(l,cv2.CV_64F,1,0, ksize=k)
            sobely = cv2.Sobel(l,cv2.CV_64F,0,1, ksize=k)
            sobelxy = cv2.Sobel(l,cv2.CV_64F,1,1,ksize=k)
            
            laplacian = cv2.Laplacian(l,cv2.CV_64F)
            fm = laplacian.var() 
            # Display
            imSx = np.dstack((sobelx,sobelx,sobelx))
            imSy = np.dstack((sobely,sobely,sobely))
            
            imSxy = np.dstack((sobelxy,sobelxy,sobelxy))
            imL   = np.dstack((laplacian,laplacian,laplacian))
            canvas= np.hstack((im,np.uint8(imL), np.uint8(imSxy), np.uint8(imSx), np.uint8(imSy) ))
            #canvas= np.hstack(( im, np.dstack((l,l,l)), np.uint8(imL) ))
            
            text  = "L Blur"
            # show the image
            cv2.putText(canvas, "{}: {:.2f}".format(text, fm), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imshow('Input | Laplacian | Sobelxy | Sobelx | Sobely', canvas)
            
            #cv2.imshow('Laplacian', laplacian)
            
            
        
        # poll keystrokes
        key = cv2.waitKey(1) & 255
        # keystroke:
        if key == 27:
            done = True
        elif key == ord(' '):#spacebar to save frame
            cv2.imwrite('../data/laplacian_im.jpg', laplacian)
            print 'image saved' 

#while
#print "Laplacian info:", type(laplacian), laplacian.dtype, laplacian.shape, laplacian.min(), laplacian.max(), im.shape
cv2.destroyAllWindows()
cap.release()