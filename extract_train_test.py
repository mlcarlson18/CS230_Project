import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.model_selection import train_test_split
from numpy import moveaxis


class extract_train_test:


    def __init__(self):
        self.mode = 1
        self.nan_pixel_value = 500

    # Convert the DICOM Pixel Data to Numpy Array while Fixing NAN
    def convert_pixel_array_to_numpy(self, x):

        # Necessary variable for numpy array intialization
        first_iteration = True
        for i in x:
            if first_iteration:
                np_i = np.asarray(i).astype(np.float32)

                # Isolate value to replace NaN with
                self.nan_pixel_value = np.nanmax(np_i.copy().flatten()) if np.nanmax(np_i.copy().flatten()) > self.nan_pixel_value else self.nan_pixel_value
                i_fixed = np.nan_to_num(np_i,
                                            nan=self.nan_pixel_value)

                to_return = i_fixed
                first_iteration = False
            else:
                np_i = np.asarray(i).astype(np.float32)

                # Isolate value to replace NaN with
                i_fixed = np.nan_to_num(np_i,
                                            nan=self.nan_pixel_value)

                to_return = np.vstack((to_return, i_fixed))
        return to_return

    # For 3D CNNs
    def brain_train_and_test(self, DATASET):
        X = list()
        y = list()
        for c in DATASET.Controls:
            r = DATASET.Rapids_per_Controls[c]
            ex, why = self.return_train_and_test_by_slice(c, r, DATASET.max_pixel_value)
            X.append(ex)
            y.append(why)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1, random_state=3)

        X_train = np.array(X_train)[:2]
        y_train = np.array(y_train)[:2]
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print("X_train Shape: ", X_train.shape)
        print("Y_train Shape: ", y_train.shape)
        print("X_test Shape: ", X_test.shape)
        print("Y_test Shape: ", y_test.shape)

        return (X_train, y_train, X_test, y_test)

    def slices_train_and_test(self, DATASET, train_set_size):
        X = list()
        y = list()
        for c in DATASET.Controls:
            r = DATASET.Rapids_per_Controls[c]
            ex, why = self.return_train_and_test_by_slice(c, r, DATASET.max_pixel_value)
            X.append(ex)
            y.append(why)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1, random_state=3)#train_size=train_set_size)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        print("X_train Shape: ", X_train.shape)
        print("Y_train Shape: ", y_train.shape)
        print("X_test Shape: ", X_test.shape)
        print("Y_test Shape: ", y_test.shape)

        # Collapse the Xs together
        X_train = X_train.reshape(-1, *X_train.shape[-3:])
        y_train = y_train.reshape(-1, *y_train.shape[-2:])
        X_test = X_test.reshape(-1, *X_test.shape[-3:])
        y_test = y_test.reshape(-1, *y_test.shape[-2:])

        print("X_train Shape: ", X_train.shape)
        print("Y_train Shape: ", y_train.shape)
        print("X_test Shape: ", X_test.shape)
        print("Y_test Shape: ", y_test.shape)

        return (X_train, y_train, X_test, y_test)

    def pixels_train_and_test(self, DATASET, train_set_size):
        X = list()
        y = list()
        for c in DATASET.Controls:
            r = DATASET.Rapids_per_Controls[c]
            ex, why = self.return_train_and_test_by_pixel(c, r,DATASET.max_pixel_value)
            X.append(ex)
            y.append(why)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state = 3)#train_size=train_set_size)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Collapse the Xs together
        X_train = X_train.reshape(-1, *X_train.shape[-1:])
        y_train = y_train.reshape(-1, *y_train.shape[-1:])
        X_test = X_test.reshape(-1, *X_test.shape[-1:])
        y_test = y_test.reshape(-1, *y_test.shape[-1:])

        print("X_train Shape: ", X_train.shape)
        print("Y_train Shape: ", y_train.shape)
        print("X_test Shape: ", X_test.shape)
        print("Y_test Shape: ", y_test.shape)

        return (X_train, y_train, X_test, y_test)

    # Extracting X and Y For CNN architectures
    def return_train_and_test_by_slice(self, control_database, rapid_database, nan_pixel_value, for_visualization_purposes = False
                                       ):

        # X and y to train our model
        X = dict()
        y = dict()

        # Iterating over Each slice (24 slice locations)
        for slice in control_database.list_of_slices:

            first_slice = True

            # RAPID file for specified slice
            y[slice] = rapid_database.DCM_per_slice_and_time[slice, 0].pixel_data

            # Iterating over each time stamp
            for time in range(control_database.number_of_timestamps - 1):

                # Add pixel data for (width, height) of specified slice for each timestamp
                if (slice, time) in control_database.DCM_per_slice_and_time:
                    pixel_data = self.convert_pixel_array_to_numpy(control_database.DCM_per_slice_and_time[slice, time].pixel_data)
                    if first_slice:
                        values = np.array(pixel_data)
                        first_slice = False
                    else:
                        values = np.vstack((values, pixel_data))
                    #values = np.append(values, self.convert_pixel_array_to_numpy_array(control_database.DCM_per_slice_and_time[slice, time].pixel_data))
                else:
                    if first_slice:
                        values = np.array(pixel_data)
                        first_slice = False
                    else:
                        values = np.vstack((values, np.full((128, 128), nan_pixel_value))
                                           )

            X[slice] = values

        X_retrieved = np.array(list(X.values())) #24, 60, 128, 128
        y_retrieved = np.array(list(y.values())) #24, 128, 128

        # Changing Shape to Desired
        X_retrieved = X_retrieved.reshape(24,60, 128, 128)
        X_retrieved = np.swapaxes(X_retrieved, 1, 2)
        X_retrieved = np.swapaxes(X_retrieved, 2, 3)

        y_retrieved = y_retrieved.reshape(24, 128, 128)

        return X_retrieved, y_retrieved




    # Divide the two databases into X and Y for running models; where X is DSC_Database, and y is Rapid_Database
    # One independent input is considered the exact pixel location of one type of slice.
    # There are 128 x 128 pixel locations, and 24 types of slices which means there are 128 x 128 x 24 = 393,218 independent inputs.
    # Each 393,218 input features consists of an array of size 60 from there being 60 timestamps
    """
    This serves as a naive benchline performance, where neighboring pixels are viewed as independent, no knowledge of
    Tmax is used in the algorithm, and only basic ML implemented
    """

    def return_train_and_test_by_pixel(self, control_database, rapid_database, nan_pixel_value, for_visualization_purposes = False):

        # Some files that don't have pixel data
        lost_files = 0

        # X and y to train our model
        X = dict()
        y = dict()

        # Iterating over Each pixel location specified by (width, height) key
        # (128 x 128) images
        for width in range(0, control_database.pixel_width):
            for height in range(0, control_database.pixel_height):

                # Iterating over Each slice (24 slice locations)
                for slice in control_database.list_of_slices:

                    # RAPID file for specified slice
                    dcm_rapid = rapid_database.DCM_per_slice_and_time[slice, 0]

                    # Add (width, height) pixel information to data for 'y'
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

        if for_visualization_purposes:
            return X

        # Convert to numpy array
        X_modified = np.array(list(X.values())) # (16384, 24, 60)
        y_modified = np.array(list(y.values())) # (16384, 24, 1)

        # Reshape to collapse pixel location and slice number
        X_modified = X_modified.reshape(-1, X_modified.shape[-1]) # (393216, 60)
        y_modified = y_modified.reshape(-1, y_modified.shape[-1]) # (393216, 1)

        # Replace nan values with max pixel value
        X_nan_fixed = np.nan_to_num(X_modified, nan=nan_pixel_value)  # SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_modified)
        Y_nan_fixed = np.nan_to_num(y_modified, nan=nan_pixel_value)  # SimpleImputer(missing_values=np.nan, strategy='mean').fit(y_modified)

        return X_nan_fixed, Y_nan_fixed

    # When trying to find vein and artery location, we want these plots to approximate our intended result
    """
    DEPRECIATED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONT USE FOR NOW 
    """
    def visualize_pixel_plot_per_slice(self, control_database, rapid_database, num_pixels, slice_index):

        # Extract X
        X = self.return_train_and_test_by_pixel(control_database, rapid_database, for_visualization_purposes=True)

        plt.figure()

        # Extracting random pixel locations to plot
        pixel_locations = [(random.randint(0,127), random.randint(0, 127)) for y in range(num_pixels)]

        for x in pixel_locations:
            # Extracting time-series for individual pixel location
            pixel_series = X[x][slice_index]

            # Plotting time-series
            plt.plot(np.arange(len(pixel_series)), pixel_series, label=x)

        # Auxiliary Plot Stuff
        suptitle = "Pixel Values over Time for Slice: " + str(slice_index)
        plt.xlabel("Time")
        plt.ylabel("Pixel Intensity Value")
        plt.title(suptitle)
        plt.show()
