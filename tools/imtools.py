#!/usr/bin/python
'''
Created on Dec 26, 2014
CURRENT FEATURES:
1. List of Images under a path
    a. All images
    b. Natural order
    c. Check and/or generate folder
    
2. Image Processing 
    a. Histogram Equalization

3. Accurate system clock timed-event

4. Rotation 
    a. Euler-Rodriguez Formula

@author: Carlos Torres <carlitos408@gmail.com>
'''


def get_imlist(path, ext = ".png"):
    """
    Returns a list of filenames for all jpg images in a directory.
    (str, str) -> (list)
    """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(ext)]
#get_imlist

def get_nat_list(path, name="file", ext = ".txt"):
    """ Returns a list of PATHS for all fils w the given sub-strings
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


def generate_folder(p="../data/"):
    """Checks if a path p exists, else generates it"""
    if os.path.isdir(p): # if the path exists check w user b4 rewritting
        print "Folder {} already exists!".format(p)
    else: # file doesnt exist. Create the folder!
        print "Creating folder: {} ".format(p)
        os.makedirs(p)
#generate_pose_folder()


def eqImg(im):
    """
    Histogram equalization.
    Input: 
        im := numpy ndarray, 1L, grayscale [0,255]
    Output:
        im:= numpy ndarray, 1L with equalized pixel intensities.
        
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


def timeEvent(t=2,d=2):
    """Creates a variable pause based on the time at which the function was 
    called. For example, function called at 1:13:50 with minute_tick: t=2 and 
    minute_delay: d=2 will pause until 1:16:00. Which is the next minute_tick 
    that meets the factors of t=2 condition and the minute_delay > 2 (d=2)"""
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



def rotate(s,theta=0,axis='x'):
    """
    Counter Clock wise (ccw) rotation of a vector s, along the <>-axis by angle theta
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

