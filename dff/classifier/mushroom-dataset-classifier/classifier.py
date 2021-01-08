# Carla de Beer
# Created: January 2021
# Feedforward neural network built with TensorFlow-Keras.
# Binary classifier, based on mushroom appearance data.
# Mushroom Dataset from: https://archive.ics.uci.edu/ml/datasets/Mushroom

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def convert_chars_to_ints(df):
    mapping_cap_shape = {'b': 1, 'c': 2, 'x': 3, 'f': 4, 'k': 5, 's': 6}
    mapping_cap_surface = {'f': 1, 'g': 2, 'y': 3, 's': 4}
    mapping_cap_color = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'r': 5, 'p': 6, 'u': 7, 'e': 8, 'w': 9, 'y': 10}
    mapping_bruises = {'t': 1, 'f': 2}
    mapping_odor = {'a': 1, 'l': 2, 'c': 3, 'y': 4, 'f': 5, 'm': 6, 'n': 7, 'p': 8, 's': 9}
    mapping_gill_attachment = {'a': 1, 'd': 2, 'f': 3, 'n': 4}
    mapping_gill_spacing = {'c': 1, 'w': 2, 'd': 3}
    mapping_gill_size = {'b': 1, 'n': 2}
    mapping_gill_color = {'k': 1, 'n': 2, 'b': 3, 'h': 4, 'g': 5, 'r': 6, 'o': 7, 'p': 8, 'u': 9, 'e': 10, 'w': 11,
                          'y': 12}
    mapping_stalk_shape = {'e': 1, 't': 2}
    mapping_stalk_root = {'b': 1, 'c': 2, 'u': 3, 'e': 4, 'z': 5, 'r': 6, '?': 0}
    mapping_stalk_surface_above_ring = {'f': 1, 'y': 2, 'k': 3, 's': 4}
    mapping_stalk_surface_below_ring = {'f': 1, 'y': 2, 'k': 3, 's': 4}
    mapping_stalk_color_above_ring = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'w': 8, 'y': 9}
    mapping_stalk_color_below_ring = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'w': 8, 'y': 9}
    mapping_veil_type = {'p': 1, 'u': 2}
    mapping_veil_color = {'n': 1, 'o': 2, 'w': 3, 'y': 4}
    mapping_ring_number = {'n': 1, 'o': 2, 't': 3}
    mapping_ring_type = {'c': 1, 'e': 2, 'f': 3, 'l': 4, 'n': 5, 'p': 6, 's': 7, 'z': 8}
    mapping_spore_print_color = {'k': 1, 'n': 2, 'b': 3, 'h': 4, 'r': 5, 'o': 6, 'u': 7, 'w': 8, 'y': 9}
    mapping_population = {'a': 1, 'c': 2, 'n': 3, 's': 4, 'v': 5, 'y': 6}
    mapping_habitat = {'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6, 'd': 7}

    df = df.replace({'cap-shape': mapping_cap_shape})
    df = df.replace({'cap-surface': mapping_cap_surface})
    df = df.replace({'cap-color': mapping_cap_color})
    df = df.replace({'bruises': mapping_bruises})
    df = df.replace({'odor': mapping_odor})
    df = df.replace({'gill-attachment': mapping_gill_attachment})
    df = df.replace({'gill-spacing': mapping_gill_spacing})
    df = df.replace({'gill-size': mapping_gill_size})
    df = df.replace({'gill-color': mapping_gill_color})
    df = df.replace({'stalk-shape': mapping_stalk_shape})
    df = df.replace({'stalk-root': mapping_stalk_root})
    df = df.replace({'stalk-surface-above-ring': mapping_stalk_surface_above_ring})
    df = df.replace({'stalk-surface-below-ring': mapping_stalk_surface_below_ring})
    df = df.replace({'stalk-color-above-ring': mapping_stalk_color_above_ring})
    df = df.replace({'stalk-color-below-ring': mapping_stalk_color_below_ring})
    df = df.replace({'veil-type': mapping_veil_type})
    df = df.replace({'veil-color': mapping_veil_color})
    df = df.replace({'ring-number': mapping_ring_number})
    df = df.replace({'ring-type': mapping_ring_type})
    df = df.replace({'spore-print-color': mapping_spore_print_color})
    df = df.replace({'population': mapping_population})
    df = df.replace({'habitat': mapping_habitat})

    return df


def create_model():
    model = Sequential()

    model.add(Dense(units=21, activation='relu', input_shape=[21, ]))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def evaluate_model():
    losses = pd.DataFrame(model.history.history)

    losses.plot()

    metrics = pd.DataFrame(model.history.history)

    metrics[['loss', 'val_loss']].plot()
    plt.savefig('images/loss-val_loss')
    plt.show()

    metrics[['accuracy', 'val_accuracy']].plot()
    plt.savefig('images/accuracy-val_accuracy')
    plt.show()

    model.evaluate(X_test, y_test, verbose=0)

    epochs = len(metrics)


df = pd.read_csv('data/mushrooms.csv')
print(df.head())

df.isnull().sum()

df = convert_chars_to_ints(df)

print(df.head())

print(df.describe().transpose())

print(df.transpose())

# Remove feature row with null values
df = df.drop(['stalk-root'], axis=1)
df.transpose()

print(list(df))

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.savefig('images/heatmap')
plt.show()

X = df.drop('class', axis=1)
y = df['class']

encoder = LabelBinarizer()

y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()

scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

model = create_model()

early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.fit(x=X_train,
          y=y_train,
          epochs=500,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])

evaluate_model()

model.save('models/mushrooms_model.h5')

predictions = np.argmax(model.predict(X_test), axis=-1)

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
