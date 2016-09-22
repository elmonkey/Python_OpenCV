#!/usr/bin/python
'''
Carlos Torres
18May2013
Camera calibration
Adapted from
	http://www.neuroforge.co.uk/index.php/77-tutorials/78-camera-calibration-python-opencv
Status: WORKING!
@author: carlos <carlitos408@gmail.com>
'''
import cv2
import cv2.cv as cv

# create an object to read images from camera 0:= first camera connected
cam = cv2.VideoCapture(0)
capture_size=(320, 240)
cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT,capture_size[1])

# chessboard pattern
dims = (8,5)
done = False
while not done:
	ret,im = cam.read()
	
	#image/frame manipulation
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# convert (np.array) to IplImage,CvMat or CvMatND
	grayIpl = cv.fromarray(gray)
	imIpl   = cv.fromarray(im)
	print type(imIpl)

	found,points = cv.FindChessboardCorners(grayIpl,dims, cv.CV_CALIB_CB_ADAPTIVE_THRESH)
	print points
	key = cv2.waitKey(5)

	if found!=0:
		cv.DrawChessboardCorners(imIpl,dims,points,found)
		# display
		cv.ShowImage("board",imIpl)
		if key == ord(' '):
			print ("saving frame as .jpg")
			cv2.imwrite("detectedChessboard.jpg",imIpl)#imIpl.save("detectedChessboard.jpg", "JPEG")

	# read keyboard
	if key == 27: # esc
		done = True

## FIRST: set matrices for collection of calibration patterns
# Number of calibration patterns used
nimageS=total_number_of_images_wIth_chessboard_pattern
# Number of points in chessboard
num_pts=width_of_board * height_of_board
opts=cv.CreateMat(nimages * num_pts, 3, cv.CV_32FC1)  # model points
ipts=cv.CreateMat((points) * num_pts, 2, cv.CV_32FC1) # matrix of model points
npts=cv.CreateMat(nimages, 1, cv.CV_32SC1)

intrinsics=cv.CreateMat(3,3,cv.CV_64FC1)
distortion=cv.CreateMat(4,1,cv.CV_64FC1)

cv.SetZero(intrinsics)
cv.SetZero(distortion)

cv.SetZero(intrinsics2)
cv.SetZero(distortion2)
# focal lengths have 1/1 ratio
intrinsics[0,0] = 1.0
intrinsics[1,1] = 1.0

## SECOND:
