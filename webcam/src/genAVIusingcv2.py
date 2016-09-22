'''
Created on May 21, 2013
Uses the webcam to generate an AVI video file
21May2013
Ref: adapted from
    http://stackoverflow.com/questions/5426637/writing-video-with-opencv-python-mac
Status: NOTWORKKING!

@author: carlos
'''
import cv2

nf = 90 # number of frames for video
# from file
path = "../img/rgbFrame.jpg"
cap  = cv2.VideoCapture(path)
if not cap:
    print "Cannot not load the image"
print type(cap)
# # from video
# cap = cv2.VideoCapture(0)
# if not cap:
#     print "Canoot not load webcam"
     
 
fps = 24
# width = int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_WIDTH))
# height= int(cv.GetCaptureProperty(cap, cv.CV_CAP_PROP_FRAME_HEIGHT))
 
width = int(cv2.cv.GetCaptureProperty(cap, cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height= int(cv2.cv.GetCaptureProperty(cap, cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
 
#uncompressed YUV 4:2:0 chroma subsampled
fourcc = cv2.cv.CV_FOURCC('I', '4', '2', '0')
writer = cv2.VideoWriter('outcv2.avi', fourcc, fps, (width, height),1)
 
for i in range (nf):
    rat, f = cap.read()
    if rat: #
        writer.write(f)