# Carla de Beer
# Created: February 2020
# A Keras-based deeplearning CNN to predict whether a cell has been infected by malaria or not.
# Based on the Udemy course: Complete TensorFlow 2 and Keras Deep Learning Bootcamp:
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp
# The project uses the malaria image dataset obtained from National Library of Medicine:
# https://lhncbc.nlm.nih.gov/publication/pub9932

import os

import matplotlib.pyplot as plt
import numpy as nps
import seaborn as sns
from matplotlib.image import imread
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = 'data/cell_images'

print('\nExamining the data ...\n')

os.listdir(data_dir)
train_path = data_dir + '/train/'
test_path = data_dir + '/test/'

os.listdir(train_path)
os.listdir(test_path)
print(os.listdir(train_path + 'parasitized')[0])

infected_cell = train_path + 'parasitized/' + os.listdir(train_path + 'parasitized')[345]
print(infected_cell)
print(imread(infected_cell).shape)
plt.imshow(imread(infected_cell))
plt.show()

uninfected_cell = train_path + 'uninfected/' + os.listdir(train_path + 'uninfected')[213]
print(imread(uninfected_cell).shape)
plt.imshow(imread(uninfected_cell))
plt.show()

print(len(os.listdir(train_path + 'parasitized/')))
print(len(os.listdir(train_path + 'uninfected/')))
print(len(os.listdir(test_path + 'uninfected/')))
print(len(os.listdir(test_path + 'uninfected/')))

print('\nCalculating dataset image dimensions ...\n')

dim1 = []
dim2 = []

for image_filename in os.listdir(train_path + 'uninfected'):
    if image_filename != '.DS_Store':
        img = imread(train_path + 'uninfected/' + image_filename)
        d1, d2, colors = img.shape
        dim1.append(d1)
        dim2.append(d2)

print(len(dim1))

sns.jointplot(dim1, dim2)
plt.show()

np.mean(dim1)
np.mean(dim2)

image_shape = (130, 130, 3)
print(130 * 130 * 3)  # = 50700 data points per image => select batches

# help(ImageDataGenerator)

imread(infected_cell).max()  # => images are already normalised

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.10,
                               height_shift_range=0.10,
                               rescale=1 / 255,  # Rescale the image by normalizing it.
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

# fill_mode deals with missing data. Fills in missing pixels with the nearest filled value.

infected_img = imread(infected_cell)
plt.imshow(infected_img)
plt.show()

plt.imshow(image_gen.random_transform(infected_img))
plt.show()

image_gen.flow_from_directory(train_path)

print('\nConstructing the model ...\n')

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

# Dropout helps reduce overfitting
model.add(Dropout(0.5))

# The last layer is binary so use sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=2)

batch_size = 16  # Powers of two; the smaller the number, the longer it takes to train

print(image_shape[:2])

print('\nCreating the generators ...\n')

# Specify here the size you want all images to take on
train_image_generator = image_gen.flow_from_directory(train_path,
                                                      target_size=image_shape[:2],
                                                      color_mode='rgb',
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      shuffle=True)

test_image_generator = image_gen.flow_from_directory(test_path,
                                                     target_size=image_shape[:2],
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='binary',
                                                     shuffle=False)

print(train_image_generator.class_indices)

# The model takes a while to train; as an alternative load the previously trained model instead
# results = model.fit_generator(train_image_generator,
#                               epochs=20,
#                               validation_data=test_image_generator,
#                               callbacks=[early_stop])

# print(model.history.history)

# model.save('models/malaria_predictor_1.h5')

print('\nLoading previous model ...\n')

model = load_model('models/malaria_predictor.h5')

model.evaluate_generator(test_image_generator)

pred = model.predict_generator(test_image_generator)
print(pred)
print(len(pred))

predictions = pred > 0.7

print('\nAnalysing the model metrics ...\n')

print(classification_report(test_image_generator.classes, predictions))

print('\nTesting with an example ...')

test_cell = test_path + 'parasitized/' + os.listdir(test_path + 'parasitized')[364]
my_loaded_image = image.load_img(test_cell, target_size=image_shape)
print(my_loaded_image)
plt.show()

my_img_array = image.img_to_array(my_loaded_image)
print(my_img_array.shape)

# Need to submit the image as part of a batch
my_img_array = np.expand_dims(my_img_array, axis=0)
print(my_img_array.shape)

print('\nProviding the prediction ...\n')

print(model.predict(my_img_array))

print('\nDONE.')
