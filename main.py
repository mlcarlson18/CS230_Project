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
import random
from DCM_Structure import DCM, DCM_DATABASE
from preprocessing import preprocessing
from CNN_models import CNN_models
from directory_manipulation import directory_operator
from datetime import date
import pandas as pd
from sklearn.model_selection import KFold

today = date.today()

############################# HYPER-PARAMETERS!!!!!!!! ##################################

standardize_pixels_between_0_and_1 = False
preprocess_type = None

# Evaluate/Visualize sklearn models with neighboring pixel independence
evaluate_sklearn_models = False
# Can try any/all of these - logistic regression is the best so far and others take a long time
sklearn_models_to_evaluate = ["KNearestNeighbor"] #["LogisticRegression"]#, "LinearRegression","SVM", "MultiLayerPerceptron"]

# Evaluates/Visualizes CNN
evaluate_CNN =  True

# Evaluates UNET
evaluate_UNET = False

CNN_model_dimension = 3
CNN_model_num_layers = 2
batch_size = 24
epoch = 25
learning_rate = 0.0001
learning_rates =  [0.00001 + random.Random(x).random() * (0.001 - 0.00001) for x in range(3)]
batch_sizes = [1 + random.Random(x + 1).random() * (37 - 1) for x in range(3)]
epochs = [10, 50]

# What has currently been implemented:
assert CNN_model_dimension == 2 or CNN_model_dimension == 3
assert CNN_model_num_layers == 1 or CNN_model_num_layers == 2

train_set_size = 0.67 # Not being implemented currently (always leaves one out for test)
k_fold_validations = 8

# If true, will save results to file for every run
save_results = True
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

    score_df = pd.DataFrame(columns=['DATE', 'BATCH_SIZE', 'LEARNING_RATE', 'EPOCHS', 'MODEL_DIMENSIONS', 'MODEL_LAYERS', 'CV', 'SCORE'])
    # Extracting X and Y
    if CNN_model_dimension == 2:
        X, y = extracter.slices_train_and_test_cross_validation(DATASET)
    elif CNN_model_dimension == 3:
        X, y = extracter.brain_train_and_test_cross_validation(DATASET)
    #Necessary for KFold
    X = np.array(X)
    y = np.array(y)

    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # Perform each cross validation evaluation
                scores = []
                kf = KFold(n_splits=k_fold_validations, random_state=4, shuffle=True)

                #Necessary for KFold
                X = np.array(X)
                y = np.array(y)

                for train_index, test_index in kf.split(X):

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)

                    if CNN_model_dimension == 2:
                        X_train = X_train.reshape(-1, *X_train.shape[-3:])
                        y_train = y_train.reshape(-1, *y_train.shape[-2:])
                        X_test = X_test.reshape(-1, *X_test.shape[-3:])
                        y_test = y_test.reshape(-1, *y_test.shape[-2:])

                    CNN_modeler = CNN_models(CNN_model_dimension, CNN_model_num_layers)
                    score = CNN_modeler.evaluate_CNN(X_train, y_train, X_test, y_test, epochs = epoch, batch_size = batch_size, learning_rate = learning_rate)
                    scores.append(score)

                result = np.mean(scores)
                score_key = "DATE: " + str(today) + "\nBATCH SIZE: " + str(batch_size) + "\nEPOCHS: " + str(epochs) +"\nLEARNING RATE: " + str(learning_rate) + "\nDIMENSIONS: " + str(CNN_model_dimension) + "\nLAYERS: " + str(CNN_model_num_layers) + "\nCROSS VALIDATIONS: " + str(k_fold_validations) + "\nSCORE: " + str(result)
                df = pd.DataFrame([today, batch_size, epochs, learning_rate, CNN_model_dimension, CNN_model_num_layers, k_fold_validations, result])
                score_df.append(df, ignore_index=True)
                print(score_key)

    # Random save file to ensure uniqueness
    save_key = int(random.Random().random() * 1000000)
    save_file = "results" + str(save_key) + ".tsv"
    score_df.to_csv(save_file, sep='\t')


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


