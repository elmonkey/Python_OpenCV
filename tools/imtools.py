#!/usr/bin/python
'''
Created on Dec 26, 2014
CURRENT FEATURES:
1. Generic
    a. All images
    b. Natural order
    c. Check/create directory
    
2. Image Processing 
    a. Histogram Equalization
    b. Moving average
    c. Median array
    d. Convert 1L uint16 (13 active bits) image to 8bit (kinect depth images)
    e. Shrinking image 
    f. Frame image
    g. Mask image
    h. Normalize image
    i. Background subtraction

3. Accurate system clock timed-event

4. Rotation 
    a. Euler-Rodriguez Formula

@author: Carlos Torres <carlitos408@gmail.com>
'''

import os
import numpy as np
import cv2


def get_nat_list(path, name="file", ext = ".txt"):
    """ 
    Returns a list of PATHS for all fils w the given sub-strings
    in NATURAL ORDER located in the given path.
    usage:
    for folder with files named: [filename0.txt, filename1.txt,... ,filenameN.txt]
    files = get_nat_list('../root/data/', 'filename','.txt') 
    (str,str,str) -> (list)
    """
    # list of paths:
    names = [os.path.join(path,f) for f in os.listdir(path) if (( f.endswith(ext) or f.endswith(ext.upper())) 
           and (f.count(name)>0 ) )]
    names = sorted(names, key=lambda x:int(x.split(name)[1].split(".")[0]))
    # images idx numbers:
    idx = sorted([ n.split(name)[1].split(".")[0] for n in names ])
    if len(names)== 0:
        print " === No images with {} under path {} were found!===".format(name, path)
    idx = np.asarray(idx,dtype=int)
    #idx.sort() # sort the indexes in ascending order
    return names, idx
#get_nat_imlist


# generate a list of image_filenames of all the images in a folder
def get_imlist(path, ext = ".png"):
    """
    Returns a non-ordered list of filenames for all jpg images in a directory.
    (str) -> ([file1.ext, file2.etx,...,fileN.ext])
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(ext)]
#get_imlist


def eqImg(im):
    """
    Image Histogram equalization.
    (1L ndarray) -> (1L ndarray)
    """
    hist, bins = np.histogram(im.flatten(), 256,[0,256])
    
    cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max()/ cdf.max()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    im = cdf[im]
    return im
#eqImg

def normalize(im, val=255):
    """
    Normalizes image by the value val to range 0,255
    (3L or 1L  uint8 ndarray) -> (3L or 1L uint8 ndarray)
    """
    im = np.uint8(im.astype(float) * 255/ val)
    return im


def movingAverage(values,window=3):
    """
    Moving average of an array (values) within a given window

    Example:
        dataset = [1,5,7,2,6,7,8,2,2,7,8,3,7,3,7,3,15,6]
        #Will print out a 3MA for our dataset
        print movingAverage(dataset,3)
    (array)(scalar) -> (array)
    """
    window = np.ones(int(window))/float(window)
    ##smas = np.convolve(values, weigths, 'valid')
    smas = np.convolve(values, window, 'same')
    return smas # as a numpy array
#movingAverage()

def fillMax(array_list):
    """
    Creates an array f from the max value of all elements at the coordinates.
    input: list of arrays, [a,b,c,...,n], 
    output: filled array, f[i,j] = max(a[i,j], b[i,j], c[i,j],...,n[i,j])
    (list[ndarray, ndarray,..., ndarray]) -> (ndarray)
    """
    for ii in xrange(len(array_list)):
        a = array_list[ii]
        # check number of channels
        if len(a.shape)>2: # 3channel images
            a = np.max(a,axis=2)
        array_list[ii] = a
    f = np.asarray(array_list)
    f = np.max(f,axis=0)
    return f
#medianArray    
    


def medianArray(array_list):
    """
    Computes the median array for list of arrays [a, b, c, ...,n]
    d[i,j] = median(a[1,j],b[1,j],c[1,j])
    (list[ndarray, ndarray,..., ndarray]) -> ndarray
    """
    for ii in xrange(len(array_list)):
        a = array_list[ii]
        # check number of channels
        if len(a.shape)>2: # 3channel images
            a = np.mean(a,axis=2)
        array_list[ii] = a
    med = np.asarray(array_list)
    med = np.median(med,axis=0)
    return med
#medianArray

def timeEvent(t=2,d=2):
    """
    Creates a variable pause based on the time at which the function was 
    called. For example, function called at 1:13:50 with minute_tick: t=2 and 
    minute_delay: d=2 will pause until 1:16:00. Which is the next minute_tick 
    that meets the factors of t=2 condition and the minute_delay > 2 (d=2)
    """
    from time import localtime#, gmtime, strftime
#     x = strftime("%a, %d %b %Y %H:%M:%S:%s", gmtime())
    c = localtime() # struct
    print 'First executed at time: %d:%d:%.2f\n' %(c.tm_hour, c.tm_min, c.tm_sec)
    ctrl = False
    done = False
    v = np.asarray(xrange(0,60,t)) # 1D array of  ticks

    while not done:
        c = localtime()
        m,s = c.tm_min, c.tm_sec/60.0 #min & seconds(converted to minutes
        dst = np.abs(v-(m+s+d))
        idx = np.argmin(dst)

        if (not ctrl):
            tick = v[idx]
            mm = tick-m-s
            print 'Time left: %d mins & %d secs. Looking for tick %d\n'%(np.floor(mm),(mm-np.floor(mm))*60 , tick)
            ctrl = True
        if m == tick:
            c = localtime()
            print 'Reached tick %dhr:%dmm:%dss\n'%(c.tm_hour, c.tm_min, c.tm_sec)
            r = 1
            k = m
            ctrl = False
            done = True
    print 'Event Timed!'
    return r,k
#timeEvent()

##def rotation_matrix(v,axis, theta):
##    """
##    Return the rotation matrix associated with counterclockwise rotation about
##    the given axis by theta radians.
##    ref: 
##        http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
##    inputs:
##        v: input vector [x,y,z]
##        axis: rotation axis e.g., [1,0,0] rotate along x-axis
##        theta: rotation angle (counterclockwise) in radians (rad=degree*pi/180) 
##    # test rotation matrix
##        v = [3,5,0]
##        axis = [4,4,1]
##        theta = 1.2
##        print (np.dot(rotation_matrix(axis,theta), v))
##        # [ 2.74911638  4.77180932  1.91629719]
##    """
##    axis = np.asarray(axis)
##    theta= np.asarray(theta)
##    axis = axis/math.sqrt(np.dot(axis,axis))
##    a = math.cos(theta/2)
##    b,c,d = -axis*math.sin(theta/2)
##    aa, bb, cc, dd = a*a, b*b, c*c, d*d
##    bc, ad, ac, ab, bd, cd, = b*c, a*d, a*c, a*b, b*d, c*d
##    matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
##                        [2*(bc-ad), aa+cc-bb-dd, 2*(bd-ac)],
##                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
##    return np.dot(matrix,v)
### rotation_matrix



def generate_folder(p="../data/"):
    """
    Checks if a directory/path p exists, else generates it
    """
    if os.path.isdir(p): # if the path exists check w user b4 rewritting
        print "Folder {} already exists!".format(p)
    else: # file doesnt exist. Create the folder!
        print "Creating folder: {} ".format(p)
        os.makedirs(p)
#generate_pose_folder()


def depth2gray(im):
    """
    Convert a 12bit (np.uint16) depthimage to a grayscale (np.uint8) image 
    with range [0 255].
    (1L uint16 ndarray) -> (1L uint8 ndarray, 3L uint8 ndarray)
    """
    im  = im.astype(float)/(2**13 -1).astype(np.uint8)
    d4d = np.dstack((im, im ,im)) # 3L image for display
    return im
#depth2gray



def rotate(s,theta=0,axis='x'):
    """
    Counter Clock wise rotation of a vector s, along the axis by angle theta
    s:= array/list of scalars. Contains the vector coordinates [x,y,z]
    theta:= scalar, <degree> rotation angle for counterclockwise rotation
    axis:= str, rotation axis <x,y,z>
    """
    theta = np.radians(theta) # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1]*np.cos(theta) - s[2]*np.sin(theta),
             s[1]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0]*np.cos(theta) + s[2]*np.sin(theta),
             s[1],
             -s[0]*np.sin(theta) + s[2]*np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1]*np.sin(theta),
             s[0] * np.sin(theta) + s[1]*np.cos(theta),
             s[2]]
    else:
        print "Error! Invalid axis rotation"
    return r
#rot_vector

def subtract_bgn(im, bgn):
    """ Subtract the backgrounbd (bgn) from the image (im). Both have the same 
    dimensions (including number of channels)"""
    # do a simple pre-filtering
    bgn = cv2.medianBlur(bgn,3)
    im  = cv2.medianBlur(im,3)
    
    fgbg = cv2.BackgroundSubtractorMOG()
#     fgmask = fgbg.apply(bgn)
    fgmask = fgbg.apply(im)
    return fgmask
# subtract_bng()
    
def remove_bed(im, bgn):
    eroelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5),(3,3)) # use for dilation
    dilelement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,20),(5,10)) # use for erosion


    diff = cv2.absdiff(bgn,im)
    t= np.uint8((diff.max()-diff.min())/2)

    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

    # Morphological filters
    diff = cv2.erode(diff,eroelement,iterations=3) #3
    diff = cv2.dilate(diff, dilelement, iterations=7) #5

    outImage = np.ones(im.shape, np.uint8)

    # Find and draw contours (when found)
    tempContours, hierarchy = cv2.findContours(diff, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE) #chain can be NONE
    for h,cnt in enumerate(tempContours):
        if len(cnt) <100:
            tempImage3 = np.ones(im.shape, np.uint8)
            #cv2.drawContours(tempImage3,[cnt], contourIdx=-1, color=(0,255,0),thickness=-1, maxLevel=0)
            cv2.drawContours(tempImage3,[cnt], contourIdx=-1, color=(255,255,255),thickness=-1)
##            tempImage4 = cv2.cvtColor(tempImage3, cv2.COLOR_BGR2GRAY)
            outImage = cv2.bitwise_and(im, tempImage3)
            cv2.drawContours(outImage,[cnt], contourIdx=-1, color=(255,0,0),thickness=1)
            cv2.imshow("output w contours", outImage)
            cv2.waitKey(0)
    #for enumerate
    return outImage
# remove_bed()




def im_diffs(im, bgn):
    """
    Computes the difference of the images and returns the difference and the 
    masked version of image (im)
    """
    diff = im - bgn
    mask = np.zeros(diff.shape, dtype = np.uint8)    
    mask[diff==0]=1
    # === APPLY MASK TO BED scene
    masked = cv2.bitwise_and(im, im, mask = mask)
    return diff, masked
#im_diffs()
    
    

def applyMask(im, mask):
    """
    Mask a gray or color image using a single channel binary mask.
    [ndarray A, ndarray B] -> ndarray C
    C.shape = A.shape
    """
    # === Normalize mask and ensure it is binary [0,1].
    mask  = (mask / mask.max()).astype(im.dtype) # binary [0,1]
    imask = np.zeros(mask.shape,dtype = im.dtype)
    imask[mask==0]=1 # inverted mask
    
    masked  = np.zeros(im.shape, dtype = im.dtype)
    imasked = np.zeros(im.shape, dtype = im.dtype)

   
    # === APPLY MASK TO BED scene
    if len(im.shape)>len(mask.shape):         # color image: 3Layer
        #print "masking 3-L."
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        masked = mask * im
    elif len(im.shape) <len(mask.shape):     #grayscale image: 1Layer
        #print "Masking 1-L."
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)        
        masked = mask * im
    elif len(im.shape) ==len(mask.shape):
        #print "Images have same N-L"
        masked = mask * im
    else: #check for some random dimension
        raise ValueError("Can only mask color or grayscale images (1L||3L)!")
    return masked#, imasked
#mask images


def cropMask(im, mask=None, th=0):
    """
    Crops mask to the areas of none-zero pixels. 
    Input:
        im - uint8 2D single channel binary[0,1] image (mask)
        th - threshold
    Output:
        imcrop - silhouette cropped image
        k - flag that indicates when something is detected in the mask 
    """
    
    if len(im.shape) > 2: 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    
    # ensure image is binary and in [0,1] range
    if mask == None:
        mask = np.zeros(im.shape, dtype=im.dtype)
        mask[im>th]=255
        rows,cols = np.nonzero(mask!=0)        
    else:
        rows,cols = np.nonzero(mask!=0)
    
    # basic check for when there is nothing to mask
    if ((not(rows.size and cols.size)) or ( (rows.min() == rows.max()) and (cols.min() == cols.max()))):
        crop = im
        k = False
        print "WARNING: Nothing to crop - returning black (empty) image"
    else:
        crop = im[rows.min():rows.max(), cols.min():cols.max()]
        k = True
    return crop, mask, k
# cropresizeMask


def deskew(im, SZ = 20, affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR):
    """
    Deskew an image using the second order moments 
    """
    var = im
    print "IM info: ", type(var), var.dtype, var.shape, var.min(), var.max(), len(np.unique(var))
    #crop image to roi:
    r, c = im.shape
    m = cv2.moments(im)
    if abs(m["mu02"]) < 1e-2:
        return im.copy()
    skew = m["mu11"]/m["mu02"]
    M  = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    im = cv2.warpAffine(im, M, (2*r, 2*c), flags=affine_flags)
    var = im
    print "IM info: ", type(var), var.dtype, var.shape, var.min(), var.max(), len(np.unique(var))
    
    return im
#deskew()


def test_deskew():
    imname = "../extracted_sleep_poses/carlos_fetalL_silhouette_bin.png"
    im = cv2.imread(imname)
    des= deskew(im[:,:,0])
    cv2.imshow("input",im)
    cv2.imshow("deskewed", des)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#def test_deskew


def shrinkIm(im, s=320):
    """
    Resize an image while maintaining its aspect ratio.
    im := image ndarray
    s  := image size (height dimension)
    
    (ndarray, scalar) -> (ndarray)
    """
    r,c= im.shape[:2]
    if r>c:
        #ratio = float(r)/c
        rfactor = float(r)/s
        new = int(float(c)/rfactor)
        im = cv2.resize(im, (new,s))
    elif r<c:
        #ratio = float(c)/r
        rfactor = float(c)/s
        new = int(float(r)/rfactor)
        im = cv2.resize(im, (s,new))        
    
    return im


def frameImage(im, s=320):
    """
    frameImage, append/pad zeros to an image to create a square-framed image.
    im := 1L or 3L image
    s  := square frame dimension
    (ndarray, scalar) -> (ndarray)
    """
    r,c = im.shape[:2]
    
    if r < s: 
        im = cv2.resize(im, (c,s))
    
    ri = int(np.ceil((s-r)/2))
    ro = ri + r
    
    ci = int(np.ceil((s-c)/2))
    co = ci + c

    if len(im.shape) == 3:
        frame = np.zeros((s,s,3), dtype = im.dtype)

        if r<c:
            print "frame: ",frame[ri:ro, :, :].shape 
            print "im: ", im.shape
            frame[ri:ro, :, :] = im  
        else:
            frame[:,ci:co,  :] = im        

    elif len(im.shape) ==2:
        frame = np.zeros((s,s), dtype = im.dtype)
        if r<c:
            frame[ri:ro,:] = im  
        else:
            frame[:,ci:co] = im
    else:
        raise ValueError ( "Invalid image type!")

    return frame
#frameImage(im)


