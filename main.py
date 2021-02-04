# pip install pydicom
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os

# Class representing one dicom file
class DCM:
    def __init__(self, filename, pixel_data = None):
        self.filename = filename
        self.pixel_data = pixel_data

# Directory of dicom files
DIRECTORY = "AX-DWI"

# List with abstracted dicom data as DCM objects
DCM_database = list()

# Traversing directory and extracting dicom files
path = os.walk(DIRECTORY)
image_files = list()
for root, directories, files in path:
    for file in files:
        # Only extract the dicom files
        if ".dcm" in file:
            image_files.append(file)

# Converting dicom files into our dicom class for future use
for filename in image_files:
    # Extract dcm data from file
    ds = dicom.dcmread(DIRECTORY + "/" + filename)

    # Create DCM class with data of interest
    d = DCM(filename,ds.pixel_array)

    # Add DCM object to our database
    DCM_database.append(d)

# List of each dicom's pixel data
all_pixel_data = [dcm.pixel_data for dcm in DCM_database]
#print(all_pixel_data)


