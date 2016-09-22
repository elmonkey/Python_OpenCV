'''
Created on May 20, 2013
Binarizes an image based on threshold "t"
20May 2013
@author: Carlos
'''
import cv2

# binarization threshold
thresh = 1
#load image in grayscale
gray = cv2.imread("../img/rgbFrame.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
# convert grayscale to binary using the outo thresh method "Otsu""
(thresh, bwauto) = cv2.threshold(gray,128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Manual thresholding
threshold = 127
bwmanual = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

# apply some morphological filters to clean the mask


# print message: 
print "press spacebar to save the binarized images"
#display
# cv2.imshow("input image", )
cv2.imshow("grayscaled image",gray)
cv2.imshow("auto binarized", bwauto)
cv2.imshow("manual binarized", bwmanual)
cv2.waitKey(0)
k = cv2.waitKey(10)
# save image
if k == ord(" "): #spacebar
    print "saving images"
    cv2.imwrite("../img/grayscale.jpg", gray)
    cv2.imwrite("../img/manualBinarized.jpg", bwmanual)
    cv2.imwrite("../img/autoBinarized.jpg", bwauto)
cv2.destroyAllWindows() 