def drawblocks(image, n=6):
    """
    Draws lines on the perimeter and over the n x n blocks. cv2 uses backwards coords
            (0,0)______(w,0)
             |           |
             |           |
             |           |
             |           |            
            (0,h)_______(w,h)
    
    input: 
        image = 1L or 3L numpy array, uint8 [0,255]
        n = block dimensions (n x n). For example, n=6 -> 6x6 = 36 blocks 
    output: image with grid dron on it 
    
    """
    # Colors
    blue = (255,0,0)
    red  = (0,0,255)
    green= (0,255,0)
    
    if len(image.shape) == 2: # check if color image, else convert it
        im = cv2.cvtColor(np.uint8(image.astype(np.float)/np.max(image)*255), cv2.COLOR_GRAY2BGR)
    elif (image.shape) >2: # images with 3Layers = color
        im = image
        
    h,w=im.shape[:2]
    
    # DRAW Perimeter                                      
    cv2.line(im, (0,0), (w-1,0),     green, thickness=4) # top  __
    cv2.line(im, (0,0), (0,h-1),     green, thickness=4) # left |
    cv2.line(im, (0,h-1), (w-1,h-1), green, thickness=4) # bottom __
    cv2.line(im, (w-1,0), (w-1,h-1), green, thickness=4) # rite |    
    
    if n > 1:
        # The lists of line intervals
        wb = np.uint8(np.floor(w/n)) # horizontal step, for vertical lines
        hb = np.uint8(np.floor(h/n)) # vertical step, for horizontal lines
        y = np.arange(wb,w,wb)
        x = np.arange(hb,h,hb)
        
        # Draw the grid
        for i in range(len(x)-1):
            cv2.line(im, (x[i], 0), (x[i], w), blue, thickness=4) # vertical lines
        for i in range(len(y)-1):
            cv2.line(im, (0, y[i]), (h, y[i]), red,  thickness=4) # horizontal lines 
    return im
#drawblocks
