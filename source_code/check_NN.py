import numpy as np
import idx2numpy
import random
import cv2 as cv
import matplotlib.pyplot as plt
from collections import deque
from NN_Model import NN_Model

raw_X_train = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-train-images-idx3-ubyte')
raw_y_train = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-train-labels-idx1-ubyte')
raw_X_test = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-test-images-idx3-ubyte')
raw_y_test = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-test-labels-idx1-ubyte')

model= NN_Model(raw_X_train = raw_X_train,
                raw_y_train = raw_y_train)

model.train()
test_res = model.test(raw_X_test = raw_X_test, raw_y_test = raw_y_test)
print(test_res)

index = random.randrange(len(raw_X_test))
model.predict(raw_X_test[55]) 