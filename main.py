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

        # How many complete scans through brain at different time points
        self.number_of_timestamps = self.derive_number_of_timestamps()

        # Shape of pixel data - assumes every image has same dimensions as first
        self.pixel_width = self.DCM_objects[0].pixel_data.shape[0]
        self.pixel_height = self.DCM_objects[0].pixel_data.shape[1]

    def getDCM_objects(self):
        return self.DCM_objects

    def size(self):
        return len(self.DCM_objects)

    def derive_number_of_timestamps(self):
        unique_timestamps = set()
        for dcm in self.DCM_objects:
            unique_timestamps.add(dcm.timestamp)
        return len(unique_timestamps)


    def get_specific_image(self, slice_location, timestamp):
        for dcm in self.DCM_objects:
            if dcm.slice_location == slice_location and dcm.timestamp == timestamp:
                return dcm

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

# Not implemented yet
def return_training_data(control_database, rapid_database):
    train_DCMs = [dcm.pixel_data for dcm in control_database.getDCM_objects()]
    train_X = dict()
    test_x = dict()

    for slice in range(slices_per_timestamp):
        train_X[slice] = dict()
        test_x[slice] = dict()
        slice_data = np.zeros((control_database.pixel_width, control_database.pixel_height))
        for time in range(control_database.number_of_timestamps):
            dcm_control = control_database.get_specific_image(slice, time)
            dcm_rapid = rapid_database.get_specific_image(slice, time)
            for width in range(control_database.pixel_width):
                for height in range(control_database.pixel_height):
                    if (width, height) in train_X[slice]:
                        train_X[slice][(width, height)].append(dcm_control.pixel_data[width][height])
                    else:
                        train_X[slice][(width, height)] = [dcm_control.pixel_data[width][height]]
                    if (width, height) in test_X[slice]:
                        test_X[slice][(width, height)].append(dcm_rapid.pixel_data[width][height])
                    else:
                        test_X[slice][(width, height)] = [dcm_rapid.pixel_data[width][height]]

    return train_X, test_X

print(return_training_data(DSC_database, RAPID_database))











