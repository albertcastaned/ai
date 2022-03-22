import pickle
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    PICKLE_FILE_NAME = "scratch_lb.pkl"
    TRESHOLD = 0.5

    def __init__(self, alpha=0.001, epochs=6000, plot=False) -> None:
        # Initialize class with pickle if found. Overriden everytime the model is fitted.
        try:
            file = open("pickles/" + self.PICKLE_FILE_NAME, "rb")
            tmp_dict = pickle.load(file)
            self.__dict__.update(tmp_dict)
        except:
            self.alpha = alpha
            self.epochs = epochs
            self.plot = plot
            self.weights = None
            self.bias = None

    # Map real value between 0 and 1
    def _sigmoid_(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        vertical_dim, horizontal_dim = x.shape

        self.weights = np.zeros(horizontal_dim)
        self.bias = 0

        loss_values = []

        for i in range(self.epochs):
            model = np.dot(x, self.weights) + self.bias
            prediction = self._sigmoid_(model)

            calculated_weight = (1 / vertical_dim) * np.dot(x.T, (prediction - y))     
            calculated_bias = (1 / vertical_dim) * np.sum(prediction - y)

            if self.plot:
                loss_values.append(np.mean(np.sum(calculated_weight)))

            self.weights -= self.alpha * calculated_weight
            self.bias -= self.alpha * calculated_bias     


        if self.plot:
            plt.plot(loss_values)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()

        self._save_pickle_()

    def _save_pickle_(self):
        file = open("pickles/" + self.PICKLE_FILE_NAME, 'wb')
        pickle.dump(self.__dict__, file, 2)
        file.close()

    def predict(self, x) -> np.ndarray:
        model = np.dot(x, self.weights) + self.bias
        numerical_predictions = self._sigmoid_(model)
        predictions = [1 if i > self.TRESHOLD else 0 for i in numerical_predictions]
        return np.array(predictions)

    def predict_probabilities(self, x) -> np.ndarray:
        model = np.dot(x, self.weights) + self.bias
        numerical_predictions = self._sigmoid_(model)
        return np.array(numerical_predictions)

    def ready(self) -> bool:
        return self.weights and self.bias
