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
    def __init__(self, DCM_objects, folder_name):

        # Folder name of DCM database
        self.folder_name = folder_name

        # Patient Identifier
        self.patient_identifier = self.folder_name.split("_")[1]

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

    def get_DCMS_per_time_series(self):
        return False


# Class representing all of the DCM objects in entire directory
class DCM_DATASET():
    def __init__(self, DCM_Database_objects):
        # List of Databases in the Class
        self.DCM_Databases = DCM_Database_objects
        # LIst of Control and Rapid Databases Respectively
        self.Controls, self.Rapids = self.segregate_databases()
        # Dictionary for retrieving corresponding folders (X and y)
        self.Controls_per_Rapids, self.Rapids_per_Controls = self.create_retrieval_dictionaries()

    def segregate_databases(self):
        controls = list()
        rapids = list()
        for database in self.DCM_Databases:
            if "PERFUSION" in database.folder_name:
                controls.append(database)
            else:
                rapids.append(database)
        return (controls, rapids)

    def create_retrieval_dictionaries(self):
        rapids_per_controls = dict()
        controls_per_rapids = dict()
        for c in self.Controls:
            for r in self.Rapids:
                if c.patient_identifier == r.patient_identifier:
                    rapids_per_controls[c] = r
                    controls_per_rapids[r] = c
        return controls_per_rapids, rapids_per_controls
