'''
Created on May 21, 2013

@author: carlos
'''
import cv2
import cv
import numpy as np

# read the video
path = "../video/out.avi"
vid = cv2.VideoCapture(path)

# read binary mask:
path = "../img/manualBinarized.jpg"
# path = "../img/grayscale.jpg"
mask = np.uint8((cv2.imread(path)/255)) # set the mask to the correct [0,1] range
#morphological filters
eroelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5),(3,3)) # use for dilation
dilelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,20),(5,10)) # use for erosion
mask       = cv2.erode(mask,eroelement,iterations=1) #3
mask       = cv2.dilate(mask, dilelement, iterations=7) #5

done = False
nf = 0

# # get some info
# _,frame = vid.read()
# result = mask*frame
# print "frame info: ",  type(frame), frame.dtype, frame.shape
# print "result info: ", type(result), result.dtype, result.shape

while not done:
    flag, frame = vid.read()
    if flag:
        result = mask*frame
        cv2.imshow("masked-avi-file", result)
        nf += 1
        # write frame
    
    k = cv2.waitKey(10)    
    if k == 27: # esc key
        done = True
        
print "The number of frames in video is %d" %(nf)
print "mask type: ", type(mask)
print "mask dimensions are: ", mask.shape
print "frame type: ", type(frame) 
print "frame dimensions are: ", np.size(frame)
cv2.destroyAllWindows()