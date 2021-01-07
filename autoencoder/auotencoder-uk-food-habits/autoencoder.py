# Carla de Beer
# Created: March 2020
# A simple autoencoder analysing eating habits within the UK.
# Based on the Udemy course: Complete TensorFlow 2 and Keras Deep Learning Bootcamp:
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

df = pd.read_csv('UK_foods.csv', index_col='Unnamed: 0')
print(df)

df.transpose()

plt.figure(figsize=(10, 10))
sns.heatmap(df)

encoder = Sequential()

encoder.add(Dense(units=17, activation='relu', input_shape=[17]))
encoder.add(Dense(units=8, activation='relu', input_shape=[8]))
encoder.add(Dense(units=4, activation='relu', input_shape=[4]))
encoder.add(Dense(units=2, activation='relu'))

decoder = Sequential()

decoder.add(Dense(units=4, activation='relu', input_shape=[2]))
decoder.add(Dense(units=8, activation='relu', input_shape=[4]))
decoder.add(Dense(units=17, activation='relu', input_shape=[8]))

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss='mse', optimizer=SGD(lr=1.5), metrics=['accuracy'])
# mse because the values are continuous

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df.transpose())
print(scaled_data.shape)
print(scaled_data)

autoencoder.fit(scaled_data, scaled_data, epochs=15)

encoded_2dim = encoder.predict(scaled_data)
print(encoded_2dim)
print(df.transpose().index)

results = pd.DataFrame(data=df.transpose().index, columns=['index'])
print(results)

encoded_df = pd.DataFrame(encoded_2dim, columns=['C1', 'C2'])
print(encoded_df)

combined = pd.concat([results, encoded_df], axis=1)
print(combined)

sns.scatterplot(x='C1', y='C2', data=combined, hue='index')
