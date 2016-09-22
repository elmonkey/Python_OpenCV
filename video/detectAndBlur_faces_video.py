#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Detect faces on the frame and blur them.

@author: carlos torres <carlitos408@gmail.com>
"""
import cv2, cv
import numpy as np

## Setup and Train the Classifier
# #Specify the trained cascade classifier
#face_cascade_name = "./haarcascade_frontalface_alt.xml"
# #Create a cascade classifier
#face_cascade = cv2.CascadeClassifier()
# #Load the specified classifier
#face_cascade.load(face_cascade_name)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

## Setup video capture:
capture_size=(640,420)  # define webcam-display dimensions
#capture_size=(1280,840)  # define webcam-display dimensions
cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, capture_size[0])
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT,capture_size[1])
ret,img = cap.read()

done = False
while not done:
    
    #Load and Preprocess the image
    # image = cv2.imread(imagepath) # for local image
    # result_image = image.copy()
    result_image = img.copy()
    if ret:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)

        #Run the classifiers
        #faces = face_cascade.detectMultiScale(grey, 1.1, 2, 0|cv2.cv.CV_HAAR_SCALE_IMAGE, (30, 30))
        #faces = face_cascade.detectMultiScale(grey, 1.3, 5)
        faces = face_cascade.detectMultiScale(
            grey,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )        
        
        print "Faces detected"

        if len(faces) != 0:         # If there are faces in the images
            for (x,y,w,h) in faces: # for each face in faces
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_grey = grey[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                cv2.imshow('img',img)
                cv2.waitKey(0)

                # get the rectangle img around all the faces
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 5)
                sub_face = img[y:y+h, x:x+w]
                # apply a gaussian blur on this new recangle image
                sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
                # merge this blurry rectangle to our final image
                result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

            cv2.imshow("Detected face", result_image)
            cv2.waitKey(0)
            done = True
##        # keystroke:
##        key = cv2.waitKey(1) & 255
##        if key ==27: #esc to exit
##            done = True
##        elif key == ord(' '):#spacebar to save frame
##            cv2.imwrite('../data/detectedFaces_im.png', result_image)
##            print 'image saved'         
##        #cv2.imwrite("./result.png", result_image)
#while
cv2.destroyAllWindows()
cap.release()