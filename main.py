# pip install pydicom
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os

# Class representing one dicom file
class DCM:
    def __init__(self, filename, pixel_data):
        self.filename = filename
        self.pixel_data = pixel_data

# Folders with ".dcm" files
DSC_DIRECTORY = "AX-DWI"
RAPID_DIRECTORY = "RAPID-TMAX"

# Databases with list of extracted ".dcm" files as DCM objects
DSC_database = list()
RAPID_database = list()

# Traverse directory and save ".dcm" filepaths into returned list
def traverse_directory(dir):
    path = os.walk(dir)
    image_files = list()
    for root, directories, files in path:
        for file in files:
            # Only extract the dicom files
            if ".dcm" in file:
                image_files.append(dir + "/" + file)
    return image_files

DSC_files = traverse_directory(DSC_DIRECTORY)
RAPID_files = traverse_directory(RAPID_DIRECTORY)


# Converting list of files into database with DCM type
def create_database(image_files):
    database = []
    for filename in image_files:
        # Extract dcm data from file
        ds = dicom.dcmread(filename)

        # Create DCM class with data of interest
        d = DCM(filename,ds.pixel_array)

        # Add DCM object to our database
        database.append(d)

    # Return database
    return database

DSC_database = create_database(DSC_files)
RAPID_database = create_database(RAPID_files)

# List of each dicom's pixel data
all_pixel_data = [dcm.pixel_data for dcm in DSC_database]
print(all_pixel_data)


