# pip install pydicom
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os
from models import models
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from DCM_Structure import DCM, DCM_DATABASE

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
def create_database(image_files, rapid=False):
    dcm_objects = []
    counter = 0 #Used to calculate which timestamp each belongs to
    for filename in image_files:

        # Extract dcm data from file
        ds = dicom.dcmread(filename)

        #Slice number - rapid does not contain same information since only one set of slices
        slice_number = counter + 1 if rapid else ds['InStackPositionNumber'].value

        # Timestamp - rapid only has one 'time' so will always be 0
        timestamp = 0 if rapid else np.round(ds.InstanceNumber / slices_per_timestamp)

        # Create DCM class with data of interest
        d = DCM(filename,ds.pixel_array, slice_number, timestamp, ds.PatientName)

        # Add DCM object to our database
        dcm_objects.append(d)

        # Update counter
        counter += 1

    # Create Database with dcm DCM_objects
    d = DCM_DATABASE(dcm_objects)
    # Return database
    return d

print("Making databases...")
DSC_database = create_database(DSC_files)
RAPID_database = create_database(RAPID_files, rapid=True)

print("Control DCM Files: ", DSC_database.size())
print("RAPID-modified DCM Files", RAPID_database.size())

# Divide the two databases into X and Y for running models; where X is DSC_Database, and y is Rapid_Database
# One independent input is considered the exact pixel location of one type of slice.
# There are 128 x 128 pixel locations, and 24 types of slices which means there are 128 x 128 x 24 = 393,218 independent inputs.
# Each 393,218 input features consists of an array of size 60 from there being 60 timestamps
"""
This serves as a naive benchline performance, where neighboring pixels are viewed as independent, no knowledge of
Tmax is used in the algorithm, and only basic ML implemented
"""
def extract_train_and_test_data(control_database, rapid_database):

    # Some files that don't have pixel data
    lost_files = 0

    # X and y to train our model
    X = dict()
    y = dict()

    # Iterating over Each pixel location specified by (width, height) key
    # (128 x 128) images
    for width in range(0,control_database.pixel_width):
        for height in range(0,control_database.pixel_height):

            # Iterating over Each slice (24 slice locations)
            for slice in control_database.list_of_slices:

                # RAPID file for specified slice
                dcm_rapid = rapid_database.DCM_per_slice_and_time[slice, 0]

                # Add (width, height) pixel information to data
                if (width, height) in y:
                    y[(width, height)].append([dcm_rapid.pixel_data[width][height]])
                else:
                    y[(width, height)] = [[dcm_rapid.pixel_data[width][height]]]

                # Control pixel values for each time point for specified slice and (width, height)
                values = []

                # Iterating over each time stamp
                for time in range(control_database.number_of_timestamps - 1):

                    # Add pixel data for (width, height) of specified slice for each timestamp
                    if (slice, time) in control_database.DCM_per_slice_and_time:
                        dcm_control = control_database.DCM_per_slice_and_time[slice, time]
                        values.append(dcm_control.pixel_data[width][height])
                    else:
                        lost_files += 1
                        values.append(float("nan"))

                # Add to X
                if (width, height) in X:
                    X[(width, height)].append(values)
                else:
                    X[(width, height)] = [values]

    lost_files /= 128 * 128

    #print("Number of lost files after converting: ", lost_files)

    # Process the X and y values

    # Convert to numpy array
    X_modified = np.array(list(X.values()))
    y_modified = np.array(list(y.values()))

    # Reshape to collapse pixel location and slice number
    X_modified = X_modified.reshape(-1, X_modified.shape[-1])
    y_modified = y_modified.reshape(-1, y_modified.shape[-1])

    # Replace nan values with 0
    X_nan_fixed = np.nan_to_num(X_modified)#SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_modified)
    Y_nan_fixed = np.nan_to_num(y_modified) #SimpleImputer(missing_values=np.nan, strategy='mean').fit(y_modified)

    return X_nan_fixed, Y_nan_fixed

# Training model
print("Extracting X and Y for SKLEARN Models...")
X, y = extract_train_and_test_data(DSC_database, RAPID_database)

# Dividing data into train/test set (later we should do cross validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# SKLEARN specific formatting
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

model_types = ["LogisticRegression", "LinearRegression","SVM"]

score_messages = []
for model_type in model_types:
    print(model_type, " running...")
    model = models(model_type) # Logistic Regression model
    model.train(X_train, y_train)
    result = model.evaluate(X_test, y_test)
    print("Result: ", result)
    score_messages.append(model_type + " | Score: " + str(result))

print(score_messages)

#print(model.cross_validate(X, y)) # Print cross evaluation score

