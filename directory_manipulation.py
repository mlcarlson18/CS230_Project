import numpy as np
import pydicom as dicom
from DCM_Structure import DCM, DCM_DATABASE, DCM_DATASET
import os
from preprocessing import preprocessing


class directory_operator:

    def __init__(self, slices_per_timestamp, preprocess_type):
        self.placeholder = True
        self.slices_per_timestamp = slices_per_timestamp
        self.preprocess_type = preprocess_type
        self.preprocesser = preprocessing()  # Preprocesser class

    def organize(self, dir):
        folders = self.traverse_group(dir)

        databases = list()
        for f in folders:
            if "TMAX" in f:
                databases.append(self.create_database(self.traverse_directory(f), f, rapid=True))
            else:
                databases.append(self.create_database(self.traverse_directory(f), f))
        return self.create_dataset(databases)

    def traverse_group(self, dir):
        path = os.walk(dir)
        dir_folders = list()
        for root, directories, files in path:
            for d in directories:
                # Only extract data folders
                if ("PERFUSION" in d or "TMAX" in d):
                    dir_folders.append(root + "/" + d)
        return dir_folders

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
    def create_database(self, image_files, folder_name, rapid=False):

        assert len(image_files) > 0

        dcm_objects = []
        counter = 0  # Used to calculate which timestamp each belongs to
        for filename in image_files:
            # Extract dcm data from file
            ds = dicom.dcmread(filename)

            # Slice number - rapid does not contain same information since only one set of slices
            slice_number = counter + 1 if rapid else ds['InStackPositionNumber'].value

            # Discount last slices when num is too hight in order to conform to 24 standard
            if slice_number > 24:
                continue

            # Timestamp - rapid only has one 'time' so will always be 0
            timestamp = 0 if rapid else np.round(ds.InstanceNumber / self.slices_per_timestamp)

            if timestamp > 60:
                continue

            # Return dicom pixel array or pre-processed version
            pixel_array = ds.pixel_array if self.preprocess_type == None else self.preprocesser.pixel_transform(
                self.preprocess_type,
                ds.pixel_array)

            # Create DCM class with data of interest
            d = DCM(filename, pixel_array, slice_number, timestamp, ds.PatientName)

            # Add DCM object to our database
            dcm_objects.append(d)

            # Update counter
            counter += 1

        # Ensuring each folder has correct number of image files
        if (rapid and len(dcm_objects) != 24) or (rapid == False and len(dcm_objects != 1440)):
            print("Problem with number of files in data folder: ", folder_name, " ", len(dcm_objects))
            return None

        # Create Database with dcm DCM_objects
        d = DCM_DATABASE(dcm_objects, folder_name)

        # Return database
        return d

    def create_dataset(self, databases):
        d = DCM_DATASET(databases)
        return d
