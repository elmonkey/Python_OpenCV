def raw_moments(im, i, j):
    """
        Raw Moments
        
        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    nx, ny = im.shape
    lx = np.arange(nx)
    ly = np.arange(ny)
    yv, xv = np.meshgrid(lx,ly, indexing='ij')
    return np.sum(im*((xv)**i)*((yv)**j))

def central_moments(im, i, j):
    """
        Central Moments
        
        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    nx, ny = im.shape
    lx = np.arange(nx)
    ly = np.arange(ny)
    yv, xv = np.meshgrid(lx,ly, indexing='ij')
    m00 = np.sum(im)
    m10 = np.sum(xv*im)
    m01 = np.sum(yv*im)
    xavg = m10/(m00)
    yavg = m01/(m00)
    xavg = np.ones([nx,ny])*xavg
    yavg = np.ones([nx,ny])*yavg
    return np.sum(im*((xv-xavg)**i)*((yv-yavg)**j))

def scale_invarient_moments(im,i,j):
    """
        Scale Invariant Moments
        
        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    return central_moments(im, i, j)/(np.sum(im)**(1+((i+j)/2)))


def hu(im):
    """
    A few scale, rotation and position invariant moment features
    input:
        im, numpy array 
    output:
        numpy array, 8-element long Hu Moment Vector
        @param im - image numpy array
    """
    # CentralMoments: Scale Invariant
    n11 = scale_invarient_moments(im, 1, 1)
    n02 = scale_invarient_moments(im, 0, 2)
    n20 = scale_invarient_moments(im, 2, 0)
    n30 = scale_invarient_moments(im, 3, 0)
    n03 = scale_invarient_moments(im, 0, 3)
    n21 = scale_invarient_moments(im, 2, 1)
    n12 = scale_invarient_moments(im, 1, 2)
    # HuMoments: Scale (Central) + Rotation Invariant
    I0 = n20 + n02
    I1 = (n20 - n02)**2+4*n11**2
    I2 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    I3 = (n30 + n12)**2 + (n21 + n03)**2
    I4 = (n30 - 3*n12)*(n30 + n12)*(((n30 + n12)**2 - 3*(n21 + n30)**2) + (3*n21 - n03)*(n21 - n03)*(3*(n30 + n12)**2 - (n21 + n30)**2))
    I5 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03)
    I6 = (3*n21 - n03)*(n30 + n21)*(((n30 + n12)**2 - 3*(n21 + n03)**2) - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2))
    I7 = n11*((n30 + n21)**2 - (n03 + n21)**2) - (n20 - n02)*(n30 + n21)*(n03 + n21)
    return np.array([I0,I1,I2,I3,I4,I5,I6,I7])

