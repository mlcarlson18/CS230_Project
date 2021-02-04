# pip install pydicom
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os

# test comment

# Class representing one dicom file
class DCM:
    def __init__(self, filename, pixel_data, slice_location, timestamp, patient_identifier):
        # Filename including directory of .dcm file
        self.filename = filename
        # 2D Array (256x256) of pixel data
        self.pixel_data = pixel_data
        # Slice location in z axis
        self.slice_location = slice_location
        # Timestamp (0...x) referring to which scan in time dcm file refers to
        self.timestamp = timestamp
        # Patient ID
        self.patient_identifier = patient_identifier

# Class for a DCM database
class DCM_DATABASE:
    def __init__(self, DCM_objects):

        self.DCM_objects = DCM_objects

        # Dictionary containing each slice locations corresponding index (for our use) Ex: (-51.33: 0, -48.23: 1, etc...)
        self.slice_index_per_location = self.derive_slice_indices_per_location()

    def getDCM_objects(self):
        return self.DCM_objects

    def size(self):
        return len(self.DCM_objects)

    # Matches every unique slice location to an index so we can use '0..23' instead of [-51.34, -48.23, ...]
    # Assumes matching slices will have exactly same location!
    def derive_slice_indices_per_location(self):
        slice_indices = dict()
        counter = 0
        for dcm in self.DCM_objects:
            slice_location = dcm.slice_location

            # Index only gets assigned once to a slice location
            if slice_location in slice_indices:
                continue
            else:
                slice_indices[slice_location] = counter
                counter += 1
        return slice_indices

    def get_DCMS_per_slice_number():
        return False

    def get_DCMS_per_time_series():
        return False

# Folders with ".dcm" files
DSC_DIRECTORY = "PERFUSION"
RAPID_DIRECTORY = "RAPID-TMAX"

#Need to generalize this - used for determining the 'timestamp' each .dcm belongs to. Seems like there's 24 images per time stamp
slices_per_timestamp = 24

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
    dcm_objects = []
    counter = 0 #Used to calculate which timestamp each belongs to
    for filename in image_files:
        # Extract dcm data from file
        ds = dicom.dcmread(filename)

        # Create DCM class with data of interest
        d = DCM(filename,ds.pixel_array, ds.SliceLocation, np.round(ds.InstanceNumber / slices_per_timestamp), ds.PatientName)

        # Add DCM object to our database
        dcm_objects.append(d)

    # Create Database with dcm DCM_objects
    d = DCM_DATABASE(dcm_objects)
    # Return database
    return d

DSC_database = create_database(DSC_files)
RAPID_database = create_database(RAPID_files)

print("Original DCM Files: ", DSC_database.size())
print("RAPID-modified DCM Files", RAPID_database.size())

# List of each dicom's pixel data in one database
all_pixel_data = [dcm.pixel_data for dcm in DSC_database.getDCM_objects()]
#print(all_pixel_data)

# Slice indices per location for one database
print(RAPID_database.slice_index_per_location)





