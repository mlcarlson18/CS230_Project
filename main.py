# pip install pydicom
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import os
from models import sklearn_models
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from extract_train_test import extract_train_test

from DCM_Structure import DCM, DCM_DATABASE
from preprocessing import preprocessing
import CNN_1x1
from directory_manipulation import directory_operator

############################# HYPER-PARAMETERS!!!!!!!! ##################################
# Evaluate/Visualize sklearn models with neighboring pixel independence
evaluate_sklearn_models = True
# Can try any/all of these - logistic regression is the best so far and others take a long time
sklearn_models_to_evaluate =  ["LogisticRegression"]#, "LinearRegression","SVM", "MultiLayerPerceptron"]

# Evaluates/Visualizes simple CNN_1x1
evaluate_CNN_1x1 = False
batch_size = 24
epochs = 500
learning_rate = 0.001

preprocess_type = None #Default is None - works well

# Assertions that hyper-parameters are correctly input (can add more)
assert (preprocess_type == None or preprocess_type == "log" or preprocess_type == "normalize")
##################################################################################################

# Import auxiliary Classes
extracter = extract_train_test() # Extracting pixel information in various formats
preprocesser = preprocessing() # Preprocesser class
directory_operator = directory_operator(slices_per_timestamp=24, preprocess_type = preprocess_type) # Directory class

# Folders with ".dcm" files
DSC_DIRECTORY = "PERFUSION"
RAPID_DIRECTORY = "RAPID-TMAX"

# Extracting ".dcm" Files per Folder
DSC_files = directory_operator.traverse_directory(DSC_DIRECTORY)
RAPID_files = directory_operator.traverse_directory(RAPID_DIRECTORY)

# Creating Database Object with Folder Data
print("Making databases...")
DSC_database = directory_operator.create_database(DSC_files)
RAPID_database = directory_operator.create_database(RAPID_files, rapid=True)

# Visualizing Time Series for set of Pixels - Not used right now, maybe later for different approach!
"""
extracter.visualize_pixel_plot_per_slice(DSC_database, RAPID_database, num_pixels=150, slice_index=12)
"""

if evaluate_CNN_1x1:

    print("Extracting X and Y For CNNs")
    X, y = extracter.return_train_and_test_by_slice(DSC_database, RAPID_database)
    CNN_1x1.evaluate_CNN(X, y, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate)

elif evaluate_sklearn_models:

    print("Extracting X and Y for SKLEARN Models...")
    X, y = extracter.return_train_and_test_by_pixel(DSC_database, RAPID_database)

    print("X Shape: ", X.shape, "Y Shape: ", y.shape)

    # Dividing data into train/test set (later we should do cross validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # SKLEARN specific formatting
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Evaluate model types and print scores
    score_messages = []
    trained_sklearn_models = dict()
    for model_type in sklearn_models_to_evaluate:

        print(model_type, " running...")

        model = sklearn_models(model_type) # Logistic Regression model
        model.train(X_train, y_train)

        trained_sklearn_models[model_type] = model

        result = model.evaluate(X_test, y_test)
        print("Result: ", result)

        score_messages.append(model_type + " | Score: " + str(result))

    print(score_messages)

    #### VISUALIZING MODEL TYPES
    for model_type in sklearn_models_to_evaluate:
        print("Visual of ", model_type)
        sklearn_models.visualize_sklearn_model(X, y, trained_sklearn_models[model_type], DSC_database)


