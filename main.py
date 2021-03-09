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
from CNN_models import CNN_models
from directory_manipulation import directory_operator

############################# HYPER-PARAMETERS!!!!!!!! ##################################

standardize_pixels_between_0_and_1 = False
preprocess_type = None

# Evaluate/Visualize sklearn models with neighboring pixel independence
evaluate_sklearn_models = False
# Can try any/all of these - logistic regression is the best so far and others take a long time
sklearn_models_to_evaluate = ["KNearestNeighbor"] #["LogisticRegression"]#, "LinearRegression","SVM", "MultiLayerPerceptron"]

# Evaluates/Visualizes simple CNN_1x1
evaluate_CNN =  False

evaluate_UNET = True

CNN_model_dimension = 3
CNN_model_num_layers = 1
batch_size = 24
epochs = 25
learning_rate = 0.0001

# What has currently been implemented:
assert CNN_model_dimension == 2 or CNN_model_dimension == 3
assert CNN_model_num_layers == 1 or CNN_model_num_layers == 2

train_set_size = 0.67 # Not being implemented currently (always leaves one out for test)
##################################################################################################

# Import auxiliary Classes
extracter = extract_train_test() # Extracting pixel information in various formats
preprocesser = preprocessing() # Preprocesser class
directory_operator = directory_operator(slices_per_timestamp=24, preprocess_type = preprocess_type) # Directory class


# Data Folder
DATA_DIRECTORY = "Data"

print("Organizing Dataset...")

# Create Organized DATASET class comprising all the MRIs
DATASET = directory_operator.organize(DATA_DIRECTORY)

# Standardize Pixels
if standardize_pixels_between_0_and_1:
    print("Standardizing Dataset b/w 0 and 1...")
    DATASET.standardize()

if evaluate_CNN:
    # Initiate CNN Class with Desired Layers and Dimension
    CNN_modeler = CNN_models(CNN_model_dimension, CNN_model_num_layers)
    print("Extracting X and Y For CNNs")

    # Divide Dataset into slices or entire brains
    if CNN_model_dimension == 2:
        X_train, y_train, X_test, y_test = extracter.slices_train_and_test(DATASET, train_set_size = train_set_size)
    elif CNN_model_dimension == 3:
        X_train, y_train, X_test, y_test = extracter.brain_train_and_test(DATASET)

    # Evaluate CNN
    CNN_modeler.evaluate_CNN(X_train, y_train, X_test, y_test, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate)

elif evaluate_UNET:
    CNN_modeler = CNN_models(0, 0, unet=True)
    X_train, y_train, X_test, y_test = extracter.brain_train_and_test(DATASET)
    CNN_modeler.evaluate_CNN(X_train, y_train, X_test, y_test, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate)

elif evaluate_sklearn_models:

    print("Extracting X and Y for SKLEARN Models...")
    X_train, y_train, X_test, y_test = extracter.pixels_train_and_test(DATASET, train_set_size = train_set_size)

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
        sklearn_models.visualize_sklearn_model(X_test, y_test, trained_sklearn_models[model_type])


