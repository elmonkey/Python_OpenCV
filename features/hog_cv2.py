#!/usr/bin/python
'''
Crude hog implementation using opencv and numpy.

@author: Carlos <carlitos408@gmail.com>
'''
import nimpy as np
import cv2

def hog(img):
    """
    Histogram of oriented gradients by Dalal and Triggs.
    (1L ndarray) -> (64-element vector)
    """
    
    # Compute the image gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists) # hist is a 64 bit vector (16 x 4)
    
    return hist
# hog
