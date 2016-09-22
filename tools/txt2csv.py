#!/usr/bin/python
""" Carlos Torres 
16Jul2013  - Read txt files from directory, extract information and write csv w new name.

Format:
Data1, Data2, Data3, LineID, ColEntryName
Ex 
    Input File:          VA10_01-3D-dpt.txt
    Output/Renamed File: VA10_01reformat.csv
Refs:
    http://docs.python.org/2/library/csv.html 
    http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory-in-python

Folders:
    Source (txfiles): txt2csv/data/txfiles
    Destination(csvfiles): txt2csv/data/csvfiles
    """

# libraries
import csv
from os import listdir
from os.path import isfile, join


# global variables: 
handlers = {'h1': 'COLUMN_ENTRY_NAME',
            'h2': 'LINE_ID',
            'h3': 'POINT_DATA',
            'h4': '1.0E8 1.0E8 1.0E8'
            }
#global variables


def renameFile(filename):
    """
    Uses txt filename to create a csv with name as described in general 
    comments
    """
    d = filename.split('-') 
    newname = d[0] + 'reformat.csv'
    return newname
#renameFile


def processFile(filename):
    """
    Opens a txt file, reads contents, and extracts the information for the 
    .csv form 
    """
    # open & read lines from file
    csvfilename = renameFile(filename)
    csvfilename = '../data/csvfiles/'+csvfilename.split('/')[3] # destination of csv
    fileContent = open(filename, 'r')
    lynes = fileContent.readlines()
    ep = False
    sp = False
    with open(csvfilename,'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(0, len(lynes)):
            lyne = lynes[i]
            lyne = lyne.strip()
            if lyne == handlers['h1']:     # check for Column_Entry_Name
                i += 1
                cen = lynes[i]
            elif lyne == handlers['h2']:   # check for Line_ID
                i += 1
                lid = lynes[i]
            elif lyne == handlers['h3']:   # check for Point_Data
                spdata = i+1               # start of point_data
                sp = True
            elif lyne == handlers['h4']:
                epdata = i                 # ends point_data
                ep = True
            if sp and ep:
                ep = False
                sp = False
                for pdlyne in lynes[spdata:epdata]:
                    pdline = pdlyne.strip()
                    datarow = pdline.split(' ')
                    datarow.append(lid.strip())
                    datarow.append(cen.strip())
                    csvrow=datarow
                    csvwriter.writerow(csvrow)
    # close files (txt & csv)
    fileContent.close()
    csvfile.close()
    return 
#processFile


## MAIN:
def main():
        
    #create a list of file paths based
    mypath = '../data/txtfiles' # location of the input files
    onlyfiles = [f for f in listdir (mypath) if isfile(join(mypath,f)) ]
    print 'Number of files in input folder: ', len(onlyfiles)

    # process the files under the given folder path:
    for filename in onlyfiles:
        processFile(mypath + '/'+filename)
        newname = renameFile(filename)
# main


if __name__=="__main__":
    main()
