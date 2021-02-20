# Class representing one dicom file
class DCM:
    def __init__(self, filename, pixel_data, slice_location, timestamp, patient_identifier):
        # Filename including directory of .dcm file
        self.filename = filename
        # 2D Array (256x256) of pixel data
        self.pixel_data = pixel_data
        # Slice location in z axis
        self.slice_location = slice_location
        # Timestamp (0...x) referring to which scan in time dcm file refers to
        self.timestamp = timestamp
        # Patient ID
        self.patient_identifier = patient_identifier


# Class for a DCM database - corresponds to one folder in the directory of DCM files
class DCM_DATABASE:
    def __init__(self, DCM_objects):

        # List of DCM objects inside database
        self.DCM_objects = DCM_objects

        # How many complete scans through brain at different time points
        self.number_of_timestamps = self.derive_number_of_timestamps()

        # Shape of pixel data - assumes every image has same dimensions as first
        self.pixel_width = self.DCM_objects[0].pixel_data.shape[0]
        self.pixel_height = self.DCM_objects[0].pixel_data.shape[1]

        # Different slices
        self.list_of_slices = self.derive_list_of_slices()

        # Dictionary that stores the DCM file corresponding to specific slice and time for easy indexing
        self.DCM_per_slice_and_time = self.derive_DCM_per_slice_and_time()

    def to_print(self):
        to_print = "DICOM DATABASE\n" + "# DICOM Objects: " + str(len(self.DCM_objects)) + \
                   "\n# Timestamps: " + str(self.number_of_timestamps) + \
                    "\n# Slices: " + str(len(self.list_of_slices))
        return to_print

    def getDCM_objects(self):
        return self.DCM_objects

    def size(self):
        return len(self.DCM_objects)

    def derive_number_of_timestamps(self):
        unique_timestamps = set()
        for dcm in self.DCM_objects:
            unique_timestamps.add(dcm.timestamp)
        return len(unique_timestamps)

    def derive_list_of_slices(self):
        unique_slices = set()
        for dcm in self.DCM_objects:
            unique_slices.add(dcm.slice_location)
        return unique_slices

    def derive_DCM_per_slice_and_time(self):
        slice_time_dictionary = dict()
        for dcm in self.DCM_objects:
            slice_time_dictionary[(dcm.slice_location, dcm.timestamp)] = dcm
        return slice_time_dictionary

    def get_DCMS_per_time_series():
        return False
