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


# Make and Compile the CNN architecture
def make_model(learning_rate):
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


# Train - Test - and Visualize the CNN output
def evaluate_CNN(X, y, epochs=500, batch_size=24, learning_rate=0.0001):
    model = make_model(learning_rate)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    score = model.evaluate(X, y)
    print("SCORE: ", score)
    output = model.predict(X)
    print("OUTPUT SHAPE: ", output.shape)
    visualize_CNN_output(output)


# Visualize the output of the CNN
def visualize_CNN_output(output):
    fig, axes = plt.subplots(4, 6)
    axes = axes.ravel()
    for i in range(24):
        axes[i].imshow(output[i])
    plt.show()
