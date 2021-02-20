import numpy as np
import pydicom as dicom
from DCM_Structure import DCM, DCM_DATABASE
import os
from preprocessing import preprocessing


class directory_operator:

    def __init__(self, slices_per_timestamp, preprocess_type):
        self.placeholder = True
        self.slices_per_timestamp = slices_per_timestamp
        self.preprocess_type = preprocess_type
        self.preprocesser = preprocessing()  # Preprocesser class

    # Traverse directory and save ".dcm" filepaths into returned list
    def traverse_directory(self, dir):
        path = os.walk(dir)
        image_files = list()
        for root, directories, files in path:
            for file in files:
                # Only extract the dicom files
                if ".dcm" in file:
                    image_files.append(dir + "/" + file)
        return image_files

    # Converting list of files into database with DCM type
    def create_database(self, image_files, rapid=False):
        dcm_objects = []
        counter = 0  # Used to calculate which timestamp each belongs to
        for filename in image_files:
            # Extract dcm data from file
            ds = dicom.dcmread(filename)

            # Slice number - rapid does not contain same information since only one set of slices
            slice_number = counter + 1 if rapid else ds['InStackPositionNumber'].value

            # Timestamp - rapid only has one 'time' so will always be 0
            timestamp = 0 if rapid else np.round(ds.InstanceNumber / self.slices_per_timestamp)

            # Return dicom pixel array or pre-processed version
            pixel_array = ds.pixel_array if self.preprocess_type == None else self.preprocesser.pixel_transform(self.preprocess_type,
                                                                                                      ds.pixel_array)

            # Create DCM class with data of interest
            d = DCM(filename, pixel_array, slice_number, timestamp, ds.PatientName)

            # Add DCM object to our database
            dcm_objects.append(d)

            # Update counter
            counter += 1

        # Create Database with dcm DCM_objects
        d = DCM_DATABASE(dcm_objects)
        # Return database
        return d
