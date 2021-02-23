"""
KNN implementation on the MNIST dataset using TensorFlow 2
"""

__author__ = "Sepehr Maleki"
__email__ = "smaleki@lincoln.ac.uk"
__date__ = "12/02/2020"


# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data

# Data
(x_train, y_train), (x_test, y_test) = load_data()
tr_idx = np.random.choice(range(0, len(x_train)), 55000, replace=False)
te_idx = np.random.choice(range(0, len(x_test)), 1000, replace=False)

x_train = x_train[tr_idx].astype('float')
y_train = y_train[tr_idx]

x_test = x_test[te_idx].astype('float')
y_test = y_test[te_idx]

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


@tf.function
def predict(xtr, xte):
    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), axis=1))
    pred = tf.argmin(distance)
    return pred

sum=0
for i in range(len(x_test)):
    nn_index = predict(x_train, x_test[i])
    if y_train[nn_index] == y_test[i]:
        sum += 1
    print("Test ", i, "Prediction: ", y_train[nn_index], "Ground-truth: ", y_test[i])

print("Done!")
print("Accuracy is: {}%".format(sum*100/len(x_test)))

