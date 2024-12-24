import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import shuffle


class Preprocess:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()

    def gender_mapping(self, mydata):
        mydata['gender'] = self.label_encoder.fit_transform(mydata['gender'])
        mydata.iloc[:, 0:5] = mydata.iloc[:, 0:5].astype(float)
        return mydata

    def category_mapping(self, y_train, y_test):
        y_train = self.one_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = self.one_encoder.transform(y_test.reshape(-1, 1))
        return y_train, y_test

    def data_scalar(self, x_train, x_test):
        x_train[:, -1] = self.scaler.fit_transform(x_train[:, -1].reshape(-1, 1)).ravel()
        x_test[:, -1] = self.scaler.transform(x_test[:, -1].reshape(-1, 1)).ravel()
        x_train[:, :] = x_train[:, :].astype(float)
        x_test[:, :] = x_test[:, :].astype(float)
        return x_train, x_test

    def train_test_split(self, x, y):
        AXTrain = x[0:30, :]
        BXTrain = x[50:80, :]
        CXTrain = x[100:130, :]
        AXTest = x[30:50, :]
        BXTest = x[80:100, :]
        CXTest = x[130:150, :]

        AYTrain = y[0:30]
        BYTrain = y[50:80]
        CYTrain = y[100:130]
        AYTest = y[30:50]
        BYTest = y[80:100]
        CYTest = y[130:150]

        x_train = np.concatenate([AXTrain, BXTrain, CXTrain], axis=0)
        x_test = np.concatenate([AXTest, BXTest, CXTest], axis=0)
        y_train = np.concatenate([AYTrain, BYTrain, CYTrain], axis=0)
        y_test = np.concatenate([AYTest, BYTest, CYTest], axis=0)

        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_test, y_test = shuffle(x_test, y_test, random_state=42)
        return x_train, x_test, y_train, y_test

    def preprocessing(self):
        mydata = pd.read_csv('birds.csv')
        # print(mydata.isnull().sum())
        mydata.fillna(mydata.ffill(), inplace=True)
        # print(mydata.isnull().sum())

        mydata = self.gender_mapping(mydata)
        print("Mapped data:", mydata)

        x = mydata.iloc[:, :-1].values
        y = mydata.iloc[:, -1].values
        # x_train, x_test, y_train, y_test = self.train_test_split(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=40, shuffle=True)
        x_train, x_test = self.data_scalar(x_train, x_test)

        y_train, y_test = self.category_mapping(y_train, y_test)

        return x_train, x_test, y_train, y_test


class Backprop:
    def __init__(self, epochsNum, addBias, learningRate, num_of_layer, num_of_neu, activation_function):
        self.weights = []
        self.bias = []
        self.nodes_output = []
        self.sigmas = []
        self.epochsNum = int(epochsNum)
        self.learningRate = float(learningRate)
        self.addBias = bool(addBias)
        self.num_of_layer = int(num_of_layer)
        self.num_of_neu = num_of_neu
        self.activation_function = int(activation_function)

    def extract_num_of_neu(self):
        neu = self.num_of_neu.split(" , ")
        number_of_neuron = []
        for n in neu:
            number_of_neuron.append(int(n))
        self.num_of_neu = number_of_neuron

    def initlaizeWeight(self):
        self.weights.clear()
        self.bias.clear()
        self.num_of_neu.insert(0, 5)
        self.num_of_neu.append(3)
        print("num_of_neu", self.num_of_neu)
        for i in range(1, self.num_of_layer + 2):
            self.weights.append(np.ones((self.num_of_neu[i], self.num_of_neu[i - 1])) * 0.001)
            if self.addBias:
                self.bias.append(np.ones((self.num_of_neu[i], 1)) * 0.001)
            else:
                self.bias.append(np.zeros((self.num_of_neu[i], 1)))

    def activeFun(self, output):
        if self.activation_function == 1:  # Sigmoid
            return scipy.special.expit(output)
        elif self.activation_function == 2:  # tanh
            return np.tanh(output)

    def dirvOfActiv(self, output):
        if self.activation_function == 1:  # Sigmoid
            return output * (1 - output)
        elif self.activation_function == 2:  # tanh
            return 1 - np.power(output, 2)

    def feedforward(self, X_train):
        # print("weights:", self.weights)
        # print("bias:", self.bias)
        self.nodes_output.clear()
        X_train = X_train.reshape(len(X_train), 1)
        fnet = np.dot(self.weights[0], X_train) + self.bias[0]
        fnetOutput = self.activeFun(fnet)
        self.nodes_output.append(fnetOutput)
        for i in range(1, self.num_of_layer + 1):
            fnet = np.dot(self.weights[i], self.nodes_output[i - 1]) + self.bias[i]
            fnetOutput = self.activeFun(fnet)
            self.nodes_output.append(fnetOutput)

    def backword(self, Y_train):
        self.sigmas.clear()
        outputLN = self.num_of_neu[-1]
        Y_train = Y_train.reshape(outputLN, 1)
        fdash = self.dirvOfActiv(self.nodes_output[self.num_of_layer])
        error_signal_of_output = (Y_train - self.nodes_output[self.num_of_layer]) * fdash
        self.sigmas.append(error_signal_of_output)
        for i in range(1, self.num_of_layer + 1):
            current_layer = self.num_of_layer - i
            # print("current Layer = ", current_layer)
            fdash = self.dirvOfActiv(self.nodes_output[current_layer])
            nextL_sigma = self.sigmas[0]
            # print("nextL_sigma = ", nextL_sigma)
            currentL_W = self.weights[current_layer + 1]
            # print("current_layer_weights = ", currentL_W)
            sigmaofcurrent = np.dot(currentL_W.T, nextL_sigma) * fdash
            self.sigmas.insert(0, sigmaofcurrent)
            # print("------------------------------------")

    def UpdateWeight(self, X):
        X = X.reshape(1, len(X))
        for i in range(self.num_of_layer + 1):
            if i == 0:
                self.weights[0] = np.add(self.weights[0], np.dot(self.learningRate * self.sigmas[0], X))
                self.bias[0] = np.add(self.bias[0], self.learningRate * self.sigmas[0])
            else:
                self.weights[i] = np.add(self.weights[i], np.dot((self.learningRate * self.sigmas[i]),
                                                                 np.array(self.nodes_output[i - 1]).reshape(1, np.array(
                                                                     self.nodes_output[i - 1]).shape[0])))
                self.bias[i] = np.add(self.bias[i], (self.learningRate * self.sigmas[i]))

    def Train(self, x_train, y_train):
        self.extract_num_of_neu()
        self.initlaizeWeight()
        for k in range(self.epochsNum):
            for j in range(len(x_train)):
                self.feedforward(x_train[j])
                self.backword(y_train[j])
                self.UpdateWeight(x_train[j])
                self.sigmas.clear()
                self.nodes_output.clear()


class Test:
    def __init__(self, x_test, y_test, weights, bias, activation_function, num_of_layer):
        self.x_test = x_test
        self.y_test = y_test
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.num_of_layer = int(num_of_layer)
        self.nodes_output = []

    def activeFun(self, output):
        if self.activation_function == 1:  # Sigmoid
            return scipy.special.expit(output)
        elif self.activation_function == 2:  # tanh
            return np.tanh(output)

    def feedforward(self, X_test):
        self.nodes_output.clear()
        X_test = X_test.reshape(len(X_test), 1)
        fnet = np.dot(self.weights[0], X_test) + self.bias[0]
        fnetOutput = self.activeFun(fnet)
        self.nodes_output.append(fnetOutput)
        for i in range(1, self.num_of_layer + 1):
            fnet = np.dot(self.weights[i], self.nodes_output[i - 1]) + self.bias[i]
            fnetOutput = self.activeFun(fnet)
            self.nodes_output.append(fnetOutput)
        return self.nodes_output[-1]

    def get_acc(self, x_train, y_train):
        y_pred = []
        for j in range(len(x_train)):
            output = self.feedforward(x_train[j])
            pred = np.argmax(output)
            y_pred.append(pred)
        y_pred = np.array(y_pred)
        true_labels = np.argmax(y_train, axis=1)

        num_classes = y_train.shape[1]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(true_labels, y_pred):
            confusion_matrix[true, pred] += 1
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(f"Train Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def test(self):
        y_pred = []

        for j in range(len(self.x_test)):
            output = self.feedforward(self.x_test[j])
            pred = np.argmax(output)
            y_pred.append(pred)
        y_pred = np.array(y_pred)
        true_labels = np.argmax(self.y_test, axis=1)

        num_classes = self.y_test.shape[1]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(true_labels, y_pred):
            confusion_matrix[true, pred] += 1
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print("\nResults:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        return confusion_matrix, accuracy
