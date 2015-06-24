def drawblocks(image, n=6, thickness=1):
    """
    Draws lines on the perimeter and over the n x n blocks. cv2 uses backwards coords
            (0,0)______(w,0)
             |           |
             |           |
             |           |
             |           |            
            (0,h)_______(w,h)
    
    input: 
        image := numpy array, 1L or 3L uint8 [0,255]
        n := int, block dimensions (n x n). For example, n=6 -> 6x6 = 36 blocks
        thickness:= int, line thickness
    output: image with grid dron on it
    """
    # Line thickness
    
    
    # Colors
    yellow =(102,255,255)
    orange =(0,128,255)
    blue = (255,0,0)
    red  = (0,0,255)
    green= (0,255,0)
    
    if len(image.shape) == 2: # check if color image, else convert it
        im = cv2.cvtColor(np.uint8(image.astype(np.float)/np.max(image)*255), cv2.COLOR_GRAY2BGR)
    elif (image.shape) >2: # images with 3Layers = color
        im = image
        
    h,w=im.shape[:2]
     
    
    if n > 0:
        # The lists of line intervals
        wb = int(np.floor(w/n)) # horizontal step, for vertical lines
        hb = int(np.floor(h/n)) # vertical step, for horizontal lines
        y = range(0,h,hb)
        x = range(0,w,wb)
        
        # Draw the grid
        color = orange
        for i in range(len(x)):
            cv2.line(im, (x[i], 0), (x[i], h), color=color,  thickness=thickness) # vertical lines
        for i in range(len(y)-1):
            cv2.line(im, (0, y[i]), (w, y[i]), color=color,  thickness=thickness) # horizontal lines

    # DRAW Perimeter
    #color=red # change the color
    cv2.line(im, (0,0), (w-1,0),     color=color, thickness=thickness) # top  __
    cv2.line(im, (0,0), (0,h-1),     color=color, thickness=thickness) # left |
    cv2.line(im, (0,h-1), (w-1,h-1), color=color, thickness=thickness) # bottom __
    cv2.line(im, (w-1,0), (w-1,h-1), color=color, thickness=thickness) # rite |   

    return im
#drawblocks
