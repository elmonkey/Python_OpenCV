#!/usr/bin/python
'''
Created on July 11, 2014
decimal2sexa.py

Auxiliary methods for decimal top sexadeciman conversions and format correction

@authors:
    Carlos Torres <carlitos408@gmail>
    Tita Torres <eryn.torres@gmail.com>
'''

def colonFormat(a,b,c,d):
    """ Takes numbers a,b,c,d and returns them in the format 'a:b:c.d'
    (num, num,num,num)->(str)
    """
    out = ':'.join([str(int(a)),str(int(b)),str(int(c))])
    out += '.'+str(round(d,2)).split('.')[-1]
    return out
# colonformat


def decimal2sexa(d):
    """converts decimal degrees to sexadecimal format
    decimal = degrees.decimal; float
    sexa    = degree:mins:secs;  string
    sexa    = [degree, mins, secs]
    (number)->(str)
    """
    degree = d
    mins   = 0.0
    secs   = 0.0
    stuff  = 0.0
    if not (type (d) == float):
        print 'input type', type(d)
        print ('warning not a float. Converting to float ')
        if (type(d) == str):
            elems = d.split('.')
            degree = float(elems[0]) + float(elems[1]/(10**(len(elems[1])))
            print 'degree', degree
        sexa = colonFormat(degree,mins,secs,stuff)
        return sexa
    # This is where the numerical conversion happens dec -> sexa
    mins = (d - int(d))*60
    secs = (mins - int(mins)) * 60
    #stuff= int(100*(secs - int(secs)))
    stuff = secs - int(secs)
    sexa = colonFormat(degree,mins,secs,stuff)
    return sexa
#decimal2sexa


### MAIN EXECUTION OF CODE
##def main():
##    # test conversion
##    d = 121.135
##    d = 153.474
##    d = 4.539
##    #d = 121
##    sexa = decimal2sexa(d)
##    print ("sexadecimal: ", sexa)
##    # test correcting formats
##    os.system ("python carmine_generate_vids.py ")
###main
##
##main()
