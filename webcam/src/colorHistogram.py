'''
Created on Nov 23, 2015
Stream integrated camera and compute frame by frame color histogram
@author: Carlos <carlitos408@gmail.com>
'''



import cv2
import numpy as np

def computeAndDraw_colorHistogram(img):
    """
    Given an3L image compute the RGB histogram.
    
        test >> img = cv2.imread('zzzyj.jpg')
    """

    h = np.zeros(img.shape)

    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)

    h=np.flipud(h)
    
    #cv2.imshow('colorhist',h)
    return hist, h
#computeAndDraw_colorHistogram()


if __name__ == "__main__":
    c = cv2.VideoCapture(0) # select the device
    # window dimensions
    w = 320
    h = 240
    c.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  w)
    c.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, h)

    done = False

    while not done:
        k = cv2.waitKey(10) & 255
        flag, frame = c.read()
        if flag:
            hist, h = computeAndDraw_colorHistogram(frame)
            cv2.imshow("webcam", frame)
            cv2.imshow('colorhist',h)
            
        # read keyboard
        if k == 27: # esc key
            print "esc key pressed. exiting!"
            done = True
        elif k == ord(' '): #spacebar
            print "spacebar pressed. saving current frame as .jpg"
            cv2.imwrite('../img/rgbFrame.jpg',f)

    cv2.destroyAllWindows()
    c.release()
