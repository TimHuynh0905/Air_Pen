import idx2numpy
import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Conv1D
from keras.optimizers import SGD
from keras import backend
import numpy as np
import matplotlib.pyplot as plt

labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
n_cat = len(labels)

class NN_Model():
    def __init__(self, raw_X_train, raw_y_train):
        self.model = keras.models.Sequential()
        self.input_shape = self.training_size(raw_X_train)
        self.X_train = self.X_data_preprocess(raw_X_train)
        self.y_train = self.y_data_preprocess(raw_y_train)

    def training_size(self, raw_X_train):
        img_height = len(raw_X_train[0])
        img_width = len(raw_X_train[1])
        input_shape = img_height*img_width
        return input_shape

    def X_data_preprocess(self, raw_X):
        X_train = raw_X.reshape(len(raw_X), self.input_shape)
        X_train = X_train.astype('float32')
        X_train /= 255
        return X_train

    def y_data_preprocess(self, raw_y):
        y_test = keras.utils.np_utils.to_categorical(raw_y)
        return y_test

    def train(self):
        self.model.add(Dense(300, input_dim = self.input_shape, activation = 'relu'))
        self.model.add(Dense(150, activation = 'relu'))
        self.model.add(Dense(300, activation = 'relu'))
        self.model.add(Dense(150, activation = 'relu'))
        self.model.add(Dense(n_cat, activation='softmax'))
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # or categorical_crossentropy
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=500)

    def test(self, raw_X_test, raw_y_test):
        X_test = self.X_data_preprocess(raw_X_test)
        y_test = self.y_data_preprocess(raw_y_test)
        results = self.model.evaluate(X_test, y_test)
        return f"loss = {results[0]*100}; accuracy = {results[1]*100}"
    
    def predict(self, Xt): # X_input = raw image size 28x28
        X = Xt.reshape(1, (Xt.shape[0]*Xt.shape[1])) # [1 x 784]
        X = X.astype('float32')  
        X /= 255
        # print(X)

        result = np.round(self.model.predict(X, verbose=1), decimals=2)
        result_label = np.argmax(result, axis=1)

        # plt.imshow(Xt.T, cmap='gray')
        # plt.title("Class {}".format(labels[result_label[0]]))
        # plt.show()
        print(labels[result_label[0]])
        return str(labels[result_label[0]])


    

