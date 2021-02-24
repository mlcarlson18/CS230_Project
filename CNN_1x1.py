import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keras
from keras import backend

"""
The idea is to use 1x1 Conv2D Layers to Collapse the 'time' dimension. In this model, we are still 
treating each slice type as independent!

We cannot use any Max Pooling or FC or Flatten layers, as those will reduce the pixel dimensions of our image
- for this - we need Unet!
"""


class CNN_models():
    def __init__(self):
        self.placeholder = True

    # Make and Compile the CNN architecture
    def make_simplest_model(self, learning_rate):

        X_input = Input((128, 128, 60))

        # Convolutional layer with 1 filter and stride = 1
        X = Conv2D(1, (1, 1), strides=(1, 1), name='conv0')(X_input)

        # Create the model
        model = Model(inputs=X_input, outputs=X, name='PleaseWork')

        # Optimizer (Will be hyperparameter during more rigorous testing)
        opt = Adam(learning_rate=learning_rate)

        # Compile model with loss function (may want to change)
        model.compile(loss='mean_squared_error', optimizer=opt)

        # Prints summary
        model.summary()

        return model

    # Slightly More Complex Architecture
    # 2 Conv2D Layers, WITH batch normalization!
    def make_medium_model(self, learning_rate):
        X_input = Input((128, 128, 60))

        X = BatchNormalization(name='bn0')(X_input)
        X = Conv2D(30, (1, 1), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(name='bn1')(X)
        X = Conv2D(1, (1, 1), strides=(1, 1), name='conv1')(X)

        # Create the model
        model = Model(inputs=X_input, outputs=X, name='TwoLayer')

        # Optimizer (Will be hyperparameter during more rigorous testing)
        opt = Adam(learning_rate=learning_rate)

        # Compile model with loss function (may want to change)
        model.compile(loss='mean_squared_error', optimizer=opt)

        # Prints summary
        model.summary()

        return model

    # Train - Test - and Visualize the CNN output
    def evaluate_CNN(self, X_train, y_train, X_test, y_test, epochs=500, batch_size=24, learning_rate=0.001):

        model = self.make_medium_model(learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        score = model.evaluate(X_test, y_test)
        print("SCORE: ", score)
        output = model.predict(X_test)
        print("OUTPUT SHAPE: ", output.shape)
        print("MODEL PREDICTION")
        self.visualize_CNN_output(output, "MODEL PREDICTION")
        print("Expected:")
        self.visualize_CNN_output(y_test, "RAPID OBSERVED RESULT")
        return score

    # Visualize the output of the CNN
    def visualize_CNN_output(self, output, title=""):
        fig, axes = plt.subplots(4, 6)
        axes = axes.ravel()
        for i in range(24):
            axes[i].imshow(output[i], cmap='gray')
        fig.suptitle(title)
        plt.show()

    def grid_search(self, X, y, learning_rates, epochs, batch_sizes):
        scores = dict()
        for lr in learning_rates:
            for e in epochs:
                for bc in batch_sizes:
                    scores[str(lr) + ":" + str(e) + ":" + str(bc)] = self.evalute_CNN(X, y, e, bc, lr)
        print(scores)
        return scores
