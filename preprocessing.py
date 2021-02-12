import numpy as np
from sklearn.preprocessing import normalize
class preprocessing:
    def __init__(self):
        True

    # Transform DICOM pixel arrays based on 'type'
    def pixel_transform(self, type, pixel_array):

        # Create copy
        new_pixel_array = np.zeros((pixel_array.shape[0], pixel_array.shape[1]))
        max_pixel_value = max(pixel_array.copy().flatten())
        min_pixel_value = min(pixel_array.copy().flatten())
        diff = max_pixel_value - min_pixel_value
        # Iterate through each pixel
        for w in range(pixel_array.shape[0]):
            for h in range(pixel_array.shape[1]):
                if type == "log":
                    new_pixel_array[w][h] = -np.log10(pixel_array[w][h] + 1e-5)
                elif type == "normalize":
                    new_pixel_array[w][h] = (pixel_array[w][h] - min_pixel_value) / diff

        # Return result
        return new_pixel_array