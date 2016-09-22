'''
Created on May 21, 2013
Uses the webcam to generate an AVI video file
21May2013
Ref: adapted from
    http://stackoverflow.com/questions/5426637/writing-video-with-opencv-python-mac
Status: WORKS- Although there are error flags (not sure why)
@author: carlos
'''

import cv
nf = 120 # number of frames for video

cap = cv.CaptureFromCAM(0) # from webcam

# video writer
fps = 24
width  = int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT))
# uncompressed YUV 4:2:0 chroma subsampled
fourcc = cv.CV_FOURCC('I','4','2','0')
writer = cv.CreateVideoWriter('../video/test.avi', fourcc, fps, (width, height), 1)

print "Press 'spacebar' to start recording avi or 'esc' to terminate"

# main loop
done = False
while not done:
    k = cv.WaitKey(10)
    cv.GrabFrame(cap)
    frame = cv.RetrieveFrame(cap)
    cv.ShowImage("video", frame)
    
    if k ==ord(" "): #spacebar
        for i in range(nf):
            cv.GrabFrame(cap)
            frame = cv.RetrieveFrame(cap)
#             cv.ShowImage("video", frame)
#             cv.WaitKey(10)
            cv.WriteFrame(writer, frame)
    if k == 27: #esc
        print "terminating program"
        done = True
    
    
