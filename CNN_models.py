import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D
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
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

"""
The idea is to use 1x1 Conv2D Layers to Collapse the 'time' dimension. In this model, we are still 
treating each slice type as independent!

We cannot use any Max Pooling or FC or Flatten layers, as those will reduce the pixel dimensions of our image
- for this - we need Unet!
"""


class CNN_models():
    def __init__(self, d, layers, unet=False):
        self.layers = layers
        self.d = d
        self.unet = unet
    # https://github.com/mrkolarik/3D-brain-segmentation/blob/master/3D-unet.py
    def model_unet(self):
        #K.set_image_data_format('channels_last')
        inputs = Input((24, 128, 128, 60))
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
        #up6 = concatenate([Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()
        # plot_model(model, to_file='model.png')

        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def model_3D_1Layer(self, learning_rate):
        X_input = Input((24, 128, 128, 60))
        X = Conv3D(1, (1,1,1), strides=(1,1,1), name='conv0')(X_input)
        model = Model(inputs=X_input, outputs=X, name="3DModel")
        opt = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer = opt)
        model.summary()
        return model

    # Make and Compile the CNN architecture
    def model_1Layer(self, learning_rate):

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
    def model_2Layer(self, learning_rate):
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
        if self.unet:
            model = self.model_unet()
        if self.d == 2:
            if self.layers == 2:
                model = self.model_2Layer(learning_rate)
            elif self.layers == 1:
                model = self.model_1Layer(learning_rate)
        elif self.d == 3:
            model = self.model_3D_1Layer(learning_rate)



        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        score = model.evaluate(X_test, y_test)
        print("SCORE: ", score)
        output = model.predict(X_test)
        print("OUTPUT SHAPE: ", output.shape)
        print("MODEL PREDICTION")

        if self.d == 2:
            self.visualize_CNN_output(output, "MODEL PREDICTION")
            print("Expected:")
            self.visualize_CNN_output(y_test, "RAPID OBSERVED RESULT")
        elif self.d == 3:
            output = output.reshape(24, 128, 128)
            self.visualize_CNN_output(output, "MODEL PREDICTION")
            print("Expected:")
            y_test = y_test.reshape(24, 128, 128)
            self.visualize_CNN_output(y_test, "RAPID OBSERVED RESULT")

        return score

    # Visualize the output of the CNN
    def visualize_CNN_output(self, output, title=""):
        fig, axes = plt.subplots(4, 6)
        axes = axes.ravel()
        for i in range(24):
            im = axes[i].imshow(output[i], cmap='gray')
        #fig.suptitle(title)
        cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(im, cax=cb_ax, shrink = 0.95)
        cbar.set_ticks([])

        plt.show()

    def grid_search(self, X, y, learning_rates, epochs, batch_sizes):
        scores = dict()
        for lr in learning_rates:
            for e in epochs:
                for bc in batch_sizes:
                    scores[str(lr) + ":" + str(e) + ":" + str(bc)] = self.evalute_CNN(X, y, e, bc, lr)
        print(scores)
        return scores

    def dice_coef(self,y_true, y_pred):
        smooth=1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
