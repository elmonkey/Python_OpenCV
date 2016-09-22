#!/usr/bin/python
'''
Created on May 23, 2013
Streams webcam and generates avi. 
    test.avi -> saved to ../video/
Simple inputs:
    ESC key terminates the program
    Spacebar captures current frame
        screenshoot_<frame number> -> saved to ../img/

Updated a6 Jun 2015. tested 6 codecs
    opencv 2.4.9
    python 2.7.6
    ubuntu 14.04 x64

@author: carlos <carlitos408@gmail.com>
'''
import cv2, cv
import numpy as np

#==============================================================================
# Video Capture Properties
#==============================================================================
cap = cv2.VideoCapture(0) # capture device
## Frame dimensions
# Auto setup -- not working
#w   = np.uint8(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
#h   = np.uint8(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
#fps = np.uint8(cap.get(cv.CV_CAP_PROP_FPS))

## Manual setup -- working
w = 640
h = 480
print "Frame dimesions w={}, h={}".format(w,h)

#==============================================================================
# Video .avi output setup
#==============================================================================
vidname= "../video/test.avi"
fps = 20.0
nf  = 90 # total number of video frames
f   = 0  # frame counter

#==============================================================================
# THE CODECS
#==============================================================================
fourcc = cv.CV_FOURCC('X', 'V', 'I', 'D') # option 1 -- preferred | smallest file
#fourcc = cv.CV_FOURCC('D','I','V','X') # option 2
#fourcc = cv.CV_FOURCC('I','4','2','0') # option 3
#fourcc = cv2.cv.CV_FOURCC('I','Y','U','V')#option 4 -- note preceeding cv2.
#fourcc = cv.CV_FOURCC('M','J','P','G') # option 5
#fourcc = -1 # option 6 -- default DOES NOT WORK!

vidout = cv2.VideoWriter(vidname, fourcc, fps, (w,h),1)
    
print "Press 'spacebar' to start recording avi or 'esc' to terminate"

done = False

while not done:
    k = cv2.waitKey(1)
    flag, frame = cap.read()
    if flag == True:
        cv2.imshow("live", frame)
        vidout.write(frame)
        f+=1
        print "frame No.",f
        if f == nf:
            done = True
        if k == 27: #esc key
            done = True
        elif k == ord(' '): #spacebar
            print 'Saving frame {}'.format(f)
            cv2.imwrite('../img/screenshoot_'+str(f)+'.png', frame)
    else:
        print "Streaming failed. Terminating!"
        break

# release resources and destoy windows
cap.release()
vidout.release()
cv2.destroyAllWindows()

print "Completed video generation using {} codec". format(fourcc)