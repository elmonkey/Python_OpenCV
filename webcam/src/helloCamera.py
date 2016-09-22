'''
Created on May 20, 2013
hello Alienware integrated camera
Carlostorres - 20May2013
@author: Carlos
'''

import cv2
import genAVI
w = 320
h = 240

c = cv2.VideoCapture(0)
c.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  w)
c.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, h)

done = False

while not done:
    k = cv2.waitKey(33)
#    key = 0xFF & cv2.waitKey(33) #this is ok
    #key = np.int16(cv2.waitKey(33)) #this is ok [2]
    flag, f = c.read()
    if flag:
        cv2.imshow("webcam", f)
    # read keyboard
    if k == 27: # esc key
        print "esc key pressed. exiting!"
        done = True
    elif k == ord(' '): #spacebar
        print "spacebar pressed. saving current frame as .jpg"
        cv2.imwrite('../img/rgbFrame.jpg',f)
    elif k == ord("r"):
        print "r key pressed. saving 100 frames of current stream"
        genAVI()
cv2.destroyAllWindows()