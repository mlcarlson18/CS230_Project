import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class extract_train_test:
    def __init__(self):
        self.mode = 1

    # Divide the two databases into X and Y for running models; where X is DSC_Database, and y is Rapid_Database
    # One independent input is considered the exact pixel location of one type of slice.
    # There are 128 x 128 pixel locations, and 24 types of slices which means there are 128 x 128 x 24 = 393,218 independent inputs.
    # Each 393,218 input features consists of an array of size 60 from there being 60 timestamps
    """
    This serves as a naive benchline performance, where neighboring pixels are viewed as independent, no knowledge of
    Tmax is used in the algorithm, and only basic ML implemented
    """

    def return_train_and_test(self, control_database, rapid_database, for_visualization_purposes = False):

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

        # Replace nan values with 0
        X_nan_fixed = np.nan_to_num(X_modified)  # SimpleImputer(missing_values=np.nan, strategy='mean').fit(X_modified)
        Y_nan_fixed = np.nan_to_num(y_modified)  # SimpleImputer(missing_values=np.nan, strategy='mean').fit(y_modified)

        return X_nan_fixed, Y_nan_fixed

    # When trying to find vein and artery location, we want these plots to approximate our intended result
    def visualize_pixel_plot_per_slice(self, control_database, rapid_database, num_pixels, slice_index):

        # Extract X
        X = self.return_train_and_test(control_database, rapid_database, for_visualization_purposes=True)

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
