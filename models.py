import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

class sklearn_models():
    def __init__(self, model_type="LogisticRegression"):
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(max_iter = 25)
        elif model_type == "LinearRegression":
            self.model = LinearRegression()
        elif model_type == "SVM":
            self.model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        elif model_type == "MultiLayerPerceptron":
            self.model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def cross_validate(self, X, y, scoring_metric='neg_mean_squared_error', cross_validation = 3):
        return cross_val_score(self.model, X, y, cv=cross_validation, scoring=scoring_metric)

    @staticmethod
    def visualize_sklearn_model(X, y, model):
        print("Visualizing Model...")

        # Sklearn specific formatting
        y = np.ravel(y)

        model_result = model.predict(X)

        # Reverting predictions back into image shapes for printing
        y_reverted = y.reshape(16384, 24, 1)
        y_reverted = y_reverted.reshape(128, 128, 24)

        model_result_reverted = model_result.reshape(16384, 24, 1)
        model_result_reverted = model_result_reverted.reshape(128, 128, 24)

        # Initializing shape of plot for model
        fig, axes = plt.subplots(4, 6)
        axes = axes.ravel()
        # Plotting Model Result
        for i in range(24):
            pixel_array = np.zeros((128, 128))
            for width in range(0, 128):
                for height in range(0, 128):
                    pixel_array[width][height] = model_result_reverted[width][height][i]
            axes[i].imshow(pixel_array, cmap='gray')
        plt.show()

        # Initializing shape of plot for RAPID
        fig, axes = plt.subplots(4, 6)
        axes = axes.ravel()
        # Plotting RAPID Result
        for i in range(24):
            pixel_array = np.zeros((128, 128))
            for width in range(0, 128):
                for height in range(0, 128):
                    pixel_array[width][height] = y_reverted[width][height][i]
            axes[i].imshow(pixel_array, cmap='gray')
        plt.show()
