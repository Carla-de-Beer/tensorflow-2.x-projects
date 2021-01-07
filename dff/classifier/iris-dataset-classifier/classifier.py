# Carla de Beer
# Created: Match 2020
# A Keras-based deep feed-forward neural network classifying iris flower species given the input parameters provided.
# Based on the Udemy course: Complete TensorFlow 2 and Keras Deep Learning Bootcamp:
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp
# The project uses the iris flower dataset obtained from the UCI Machine Learning Repository:
# https://archive.ics.uci.edu/ml/datasets/Iris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import joblib

epochs = 500
patience = 3


def read_data():
    iris = pd.read_csv('data/iris.csv')
    iris.head()

    return iris


def create_X_y(iris):
    X = iris.drop('species', axis=1)
    y = iris['species']

    return X, y


def create_data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test


def scale_data(scaler):
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test


def create_model():
    model = Sequential()

    model.add(Dense(units=4, activation='relu', input_shape=[4, ]))
    # model.add(Dropout(0.2))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaulate_metrics(model):
    metrics = pd.DataFrame(model.history.history)

    metrics[['loss', 'val_loss']].plot()
    plt.savefig('images/loss-val_loss')
    plt.show()

    metrics[['accuracy', 'val_accuracy']].plot()
    plt.savefig('images/accuracy-val_accuracy')
    plt.show()


def return_prediction(model, scaler, sample_json):
    sepal_length = sample_json['sepal_length']
    sepal_width = sample_json['sepal_width']
    petal_length = sample_json['petal_length']
    petal_width = sample_json['petal_width']

    flower = [[sepal_length, sepal_width, petal_length, petal_width]]
    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_index = model.predict_classes(flower)

    return classes[class_index[0]]


# #########################################################
# Build and run the neural network
# #########################################################

iris = read_data()

X, y = create_X_y(iris)
y.unique()

# Binarise labels in a one-vs-all fashion (one-hot encoding)
encoder = LabelBinarizer()

y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = create_data_split(X, y)

scaler = MinMaxScaler()

scaled_X_train, scaled_X_test = scale_data(scaler)

model = create_model()

early_stop = EarlyStopping(monitor='val_loss', patience=patience)

model.fit(x=scaled_X_train,
          y=y_train,
          epochs=epochs,
          validation_data=(scaled_X_test, y_test),
          callbacks=[early_stop])

evaulate_metrics(model)

# Return the loss and accuracy figures from the last epoch
model.evaluate(scaled_X_test, y_test, verbose=0)

# Accuracy is high enough, now retraining on all the data
scaled_X = scaler.fit_transform(X)

# Redefine the model
model = create_model()

model.fit(scaled_X, y, epochs=epochs)
model.save('model/final_iris_model.h5')

# Save scaler for later use external to this program
joblib.dump(scaler, 'model/iris_scaler.pkl')

flower_model = load_model('model/final_iris_model.h5')
flower_scaler = joblib.load('model/iris_scaler.pkl')

iris.head()

flower_example1 = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
flower_example2 = {"sepal_length": 2.1, "sepal_width": 5.5, "petal_length": 1.4, "petal_width": 3.2}

predicted_index1 = return_prediction(model, scaler, flower_example1)
predicted_index2 = return_prediction(model, scaler, flower_example2)

print("Prediction 1:")
print(predicted_index1)

print("Prediction 2:")
print(predicted_index1)
