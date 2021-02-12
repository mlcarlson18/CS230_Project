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
from extract_train_test import extract_train_test

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

# Size of databases
print("Control DCM Files: ", DSC_database.size())
print("RAPID-modified DCM Files", RAPID_database.size())


"""
#Print a specific DICOM image from database
plt.figure()
print(RAPID_database.DCM_objects[18].pixel_data)
plt.imshow(RAPID_database.DCM_objects[18].pixel_data)
plt.show()
"""

# Extracter class with built-in functions for visualizing and extracting information from database
extracter = extract_train_test()

# Visualizing Time Series for set of Pixels
"""
extracter.visualize_pixel_plot_per_slice(DSC_database, RAPID_database, num_pixels=150, slice_index=12)
"""

print("Extracting X and Y for SKLEARN Models...")
X, y = extracter.return_train_and_test(DSC_database, RAPID_database)

print("X Shape: ", X.shape, "Y Shape: ", y.shape)

# Dividing data into train/test set (later we should do cross validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# SKLEARN specific formatting
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Model types for evaluating model
model_types = ["LogisticRegression", "LinearRegression","SVM", "MultiLayerPerceptron"]

# Evaluate several models performance
evaluate_models = False

# Visualize one models performance with images
visualize_model = True

# Evaluate performance of several models
if evaluate_models:
    score_messages = []
    for model_type in model_types:
        print(model_type, " running...")
        model = models(model_type) # Logistic Regression model
        model.train(X_train, y_train)

        result = model.evaluate(X_test, y_test)
        print("Result: ", result)
        score_messages.append(model_type + " | Score: " + str(result))

    print(score_messages)

# Visualize a specific model's performance against ground truth
# Graphs saved as .png file in current directory
if visualize_model:

    # Database that was used (not necessary to include this)
    control_database = DSC_database

    # Model for visualizing
    model = models("LogisticRegression")

    # Sklearn specific formatting
    y = np.ravel(y)

    # Train model and return prediction
    model.train(X, y)
    model_result = model.predict(X)

    # Reverting predictions back into image shapes for printing
    y_reverted = y.reshape(16384, 24, 1)
    y_reverted = y_reverted.reshape(128, 128, 24)

    model_result_reverted = model_result.reshape(16384, 24, 1)
    model_result_reverted = model_result_reverted.reshape(128, 128, 24)

    # Initializing shape of plot
    fig, axes = plt.subplots(4,6)
    axes = axes.ravel()
    # Plotting Model Result
    for i in range(24):
        pixel_array = np.zeros((control_database.pixel_width,control_database.pixel_height))
        for width in range(0,control_database.pixel_width):
             for height in range(0,control_database.pixel_height):
                pixel_array[width][height] = model_result_reverted[width][height][i]
        axes[i].imshow(pixel_array)
    plt.savefig("model_results.png")

    # Initializing shape of plot
    fig, axes = plt.subplots(4,6)
    axes = axes.ravel()
    # Plotting RAPID Result
    for i in range(24):
        pixel_array = np.zeros((control_database.pixel_width,control_database.pixel_height))
        for width in range(0,control_database.pixel_width):
             for height in range(0,control_database.pixel_height):
                pixel_array[width][height] = y_reverted[width][height][i]
        axes[i].imshow(pixel_array)
    plt.savefig("rapid_results.png")

#print(model.cross_validate(X, y)) # Print cross evaluation score

