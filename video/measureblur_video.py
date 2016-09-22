# -*- coding: utf-8 -*-
"""
Prototyping and testing blur measures
1. Marziliano's Algorithm from ICIP 2002
2. Laplace Variance algorithm from Pech2000

3. WIP: Execute on a tile-basis

Created September 2015

@author: Carlos Torres <carlos.torres@caugnate.com>
"""

import numpy as np
import cv2
    
def getTiles(src,wb=0,hb=0,thickness=1):
    """
    Tile an image into t-tiles using the follwing order.
    dims=(3,3) gives:
        |t1|t2|t3|
        |t4|t5|t6|
        |t7|t8|t9|
    
    input: 
        src := numpy array, 1L
        wb, hb := int, tiling dimensions (nx x ny). Example, 6,6 = 36 tiles
        thickness:= int, line thickness
        
    output: 
        tiles, dictionary of image with grid overlayed and individual tiles
            tiles['0']   := image with overlayed grid
            tiles['1']   := tile_1
            tiles[str(N)]:= tile_N
    """    
    # Color
    green= (0,255,0)
    
    # rgb color image for drawing blocks
    im = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    h,w=src.shape
    
    # The intervals
    y = np.linspace(0,h,wb+1,dtype=int)
    x = np.linspace(0,w,hb+1,dtype=int)
    
    r = 1 # rows counter
    c = 1 # cols counter
    t=1
    tiles={}
    for j in range(len(y)-1):    
        for i in range(len(x)-1):
            tiles[str(t)] = src[y[j]:y[c],x[i]:x[r]] #save tile to dictionary
            #cv2.imshow("Block {}".format(count), b[str(count)])
            t+=1
            r+=1
        r=1            
        c+=1
    c=1

    # Draw the grid
    color = green
    for i in x:
        cv2.line(im, (i, 0), (i, h), color=color,  thickness=thickness) # vertical lines
    for j in y:
        cv2.line(im, (0, j), (w, j), color=color,  thickness=thickness) # horizontal lines

    # DRAW Perimeter
    cv2.line(im, (0,0), (w-1,0),     color=color, thickness=thickness) # top  __
    cv2.line(im, (0,0), (0,h-1),     color=color, thickness=thickness) # left |
    cv2.line(im, (0,h-1), (w-1,h-1), color=color, thickness=thickness) # bottom __
    cv2.line(im, (w-1,0), (w-1,h-1), color=color, thickness=thickness) # rite |    
    
    tiles['0'] = im
    return tiles
#drawblocks




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


def lowPass(src):
    """
    Low pass the given src ndarray using gaussian blur.
    """
    blur = cv2.GaussianBlur(src,(5,5),0.4)
    return blur
#lowPass


def varianceOfLaplacian(src):
    """
    Viariance of laplacian
    Input:
        src, 1L ndarray
            L channel for rgb converted images or grayscale
    Output:
        float, blur measure based on the variance of the laplace response. 
               blur = var = stddev**2 = ( E[x**2] - (E[x]**2) )**2
    """
    laplacian = cv2.Laplacian(src,cv2.CV_64F)
    blur= laplacian.var() 
    return blur
#varianceOfLaplacian


def marzilianoBlur(src, width=11):
    """
    Python implementation of marziliano blur measure scanning rows for edges of
    width = width.
    
    Notes: Marziliano uses width=11 for 24 bit rgb images of 768 x 512 pixels. 
           and only uses vertical edges. 
           The paper lacks implementation details. 
           Figure2 is onconsistent with text.
           Width value is is not explained! Width depends on image size and 
           sensor-to-scene distance.
           A blur threshold is not established just a scribble of a measure.
    Input:
        src, uint8 1L ndarray grayscale or L-channel from converted rgb images.
        with, int width in pixels for edges. 
    Output:
        sobel, 1L uint8
    """        
    #edges  = 0
    edges_width=[]
    im_average_pixel=[] # Intensity average of pixels indexed by edge locations
    
    #smedges= 0
    smedges_width=[]
    smedges_average=[]
    
    count  = 0
    ## Use uint8 to detect zeros
    sobel = cv2.Sobel(src, cv2.CV_8U, 1, 0, ksize=5)
    sobel[sobel!=0]=1 
    ## Get image dimensions
    rows, cols = sobel.shape
    
    for row in range(rows):
        for col in range(cols):
            if sobel[row,col] == 1:
                if count == 0:
                    # save start of edge
                    col_start = col
                count += 1
            else: #sobel[row,col]=0
                if count != 0:
                    col_end = col # save end of edge
                    if 0 < count < width:
                        smedges_width.append(count)
                    elif count == width:
                        edges_width.append(count)
                        average_pixel = np.mean(src[row, col_start:col_end])
                        im_average_pixel.append(average_pixel)
                    else: #count > width
                        print "really wide edges"
                    count = 0
    widths_list= smedges_width + edges_width
    blur = np.mean(np.array(im_average_pixel))
    #print "The image has Marziliano blur measure = {}".format(blur)
    
    return blur
    


if __name__ == "__main__":
    marz=[] # marziliano
    voL =[] # variance of Laplace
    
    cap = cv2.VideoCapture(1)
    ret,im = cap.read()
    wb=2
    hb=1
    ts=wb*hb # number of tiles
    if ret:
        prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        done = False

        while not done:
         
            ## Get frame
            ret,im = cap.read()
            if ret:
                lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
                l = lab[:,:,0]
            
            ## Keystrokes
            key = cv2.waitKey(1) & 255
            if key == 27:
                done = True
            elif key == ord(' '):#spacebar to save frame
                cv2.imwrite('../data/laplacian_im.jpg', laplacian)
                print 'image saved'

            ## Pre-process frame:
            # Step1: Histogram Eqaulization
            l = eqImg(l)
            #Step2: LowPass to remove noise
            l = lowPass(l)
            
            ## Two global methods:
            # Call variance of laplaician method        
            voL.append(varianceOfLaplacian(l))
            # Call marziliano's method
            marz.append(marzilianoBlur(l))
            
            
            tiles = getTiles(l,wb=wb, hb=hb)
            if ts> 1:
                for t in range(1,ts+1):
                    tile = tiles[str(t)]
                    cv2.imshow('Tile_{} '.format(t), tile)
                    voL.append(varianceOfLaplacian(tile))
                    marz.append(marzilianoBlur(tile))
            #cv2.imshow('Webcam: L', np.dstack((l,l,l)))                
            cv2.imshow('Webcam:', im)
            cv2.waitKey(0)
            done = True
    
            
    #while
    # print "Laplacian info:", type(laplacian), laplacian.dtype, laplacian.shape, laplacian.min(), laplacian.max(), im.shape
    print "Marziliano blur for the whole image = {}. Tiles (t1,t2,...,tN) = {}".format(marz[0], marz[0:])
    print "varianceOfLaplace blur for the whole image = {}. Tiles (t1,t2,...,tN) = {}".format(voL[0], voL[0:])
    cv2.destroyAllWindows()
    cap.release()