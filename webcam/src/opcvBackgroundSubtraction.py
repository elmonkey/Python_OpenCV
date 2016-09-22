'''
Created on May 20, 2013
Ref:
http://blog.thoughtfultech.co.uk/blog/2012/04/09/some-simple-background-subtraction-for-loday/

but modified to conform with module cv2
20May2013
@author: carlos
'''
import cv2
import cv2.cv as cv
import numpy as np


fps = 25.0
w = 320
h = 240
im = cv2.imread("../img/rgbFrame.jpg")
inBackground = cv2.cvtColor(im, cv.CV_BGR2GRAY)
path = "../video/test.avi"
vid      = cv2.VideoCapture(0)
# writer   = cv2.VideoWriter("../video/subtracted.avi", cv.CV_FOURCC('M', 'P', '4', '2'), fps, inBackground.shape,1)
# writer     = cv2.VideoWriter("../video/subtracted.avi", cv.CV_FOURCC('I', '4', '2', '0'), fps, inBackground.shape,1)
fourcc = cv.CV_FOURCC('D','I','V','X')
writer = cv2.VideoWriter('../video/subtracted.avi', fourcc, fps, (w, h), 1)
# tempVid3     = cv2.VideoWriter("../video/out4.avi", cv.CV_FOURCC('M', 'P', '4', '2'), fps, inBackground.shape,1)
# print type(inBackground), inBackground.dtype, inBackground.shape
if vid:
    print "read"

eroelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5),(3,3)) # use for dilation
dilelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,20),(5,10)) # use for erosion

# aux vars
nf_total= 0
nf_used = 0
done = False
record = False
#main loop
while not done:
    k = cv2.waitKey(10)
    if k == 27: #esc --> terminate
        done = True
        
    flag,frame = vid.read()
    if flag: # check if frame is present
        cv2.imshow("live stream", frame)
        tempImage1 = cv2.cvtColor(frame,cv.CV_BGR2GRAY)
        tempImage2 = cv2.absdiff(inBackground,tempImage1)
        th, tempImage2 = cv2.threshold(tempImage2, 30, 255, cv2.THRESH_BINARY)
        
        # Morphological filters
        tempImage2 = cv2.erode(tempImage2,eroelement,iterations=3) #3
        tempImage2 = cv2.dilate(tempImage2, dilelement, iterations=7) #5

#         cv2.imshow("tImage2", tempImage2)
        outImage = np.ones(im.shape, np.uint8)

        # Find and draw contours (when found)
        tempContours, hierarchy = cv2.findContours(tempImage2, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE) #chain can be NONE
        for h,cnt in enumerate(tempContours):
            if len(cnt) >100:
#                 print "passed cnt"
                tempImage3 = np.ones(im.shape, np.uint8)
    #             cv2.drawContours(tempImage3,[cnt], contourIdx=-1, color=(0,255,0),thickness=-1, maxLevel=0)
                cv2.drawContours(tempImage3,[cnt], contourIdx=-1, color=(255,255,255),thickness=-1)
                tempImage4 = cv2.cvtColor(tempImage3, cv.CV_BGR2GRAY)
                outImage = cv2.bitwise_and(frame, tempImage3)
                cv2.drawContours(outImage,[cnt], contourIdx=-1, color=(255,0,0),thickness=1)
                cv2.imshow("output w contours", outImage)
                cv2.waitKey(5)
                nf_used +=1
                writer.write(outImage)
        #for enumerate
        nf_total+=1
    #if flag
#while done
print "No. of frames in video", nf_total
print "No. of frames used ", nf_used
cv2.destroyAllWindows()
