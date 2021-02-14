import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from data_handler import DataHandler
from util_provider import UtilProvider

num_classes = 2
epochs = 50
patience = 15

image_size = 256
batch_size = 16
img_height = image_size
img_width = image_size

image_shape = (image_size, image_size, 3)
print(image_shape[:2])


def set_tensorboard_config():
    return TensorBoard(log_dir=log_dir,
                       histogram_freq=1,
                       write_graph=True,
                       write_images=True,
                       update_freq='epoch',
                       profile_batch=2,
                       embeddings_freq=1)


def display_tensorboard_images(log_dir):
    file_writer = tf.summary.create_file_writer(log_dir)
    max_outputs = 12

    with file_writer.as_default():
        tf.summary.image('Training Data 1: Step 1', train_generator[0][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 1: Step 10', train_generator[0][0], step=10, max_outputs=max_outputs)
        tf.summary.image('Training Data 2: Step 1', train_generator[1][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 2: Step 10', train_generator[1][0], step=10, max_outputs=max_outputs)
        tf.summary.image('Training Data 3: Step 1', train_generator[2][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 3: Step 10', train_generator[2][0], step=10, max_outputs=max_outputs)
        tf.summary.image('Training Data 3: Step 1', train_generator[3][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 3: Step 10', train_generator[3][0], step=10, max_outputs=max_outputs)
        tf.summary.image('Training Data 4: Step 1', train_generator[4][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 4: Step 10', train_generator[4][0], step=10, max_outputs=max_outputs)
        tf.summary.image('Training Data 5: Step 1', train_generator[5][0], step=1, max_outputs=max_outputs)
        tf.summary.image('Training Data 5: Step 10', train_generator[5][0], step=10, max_outputs=max_outputs)


def run_tensorboard(log_dir):
    display_tensorboard_images(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    set_tensorboard_config()

    return tensorboard_callback


def create_model():
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Regularisation helps reduce overfitting
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    # Dropout helps reduce overfitting
    model.add(Dropout(0.7))

    # The last layer is binary so use sigmoid
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model


def fit_model():
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)

    return model.fit(train_generator,
                     validation_data=validation_generator,
                     steps_per_epoch=train_steps_per_epoch,
                     epochs=epochs,
                     validation_steps=val_steps_per_epoch,
                     verbose=2,
                     callbacks=[early_stop, tensorboard_callback])


# Run code

# Get and process data
DataHandler.extract_data_files()

num_images = DataHandler.calculate_num_images()
train_steps_per_epoch = np.ceil((num_images * 0.8 / batch_size) - 1)
val_steps_per_epoch = np.ceil((num_images * 0.2 / batch_size) - 1)

train_dir = './data/train/'

# Print image join plots
UtilProvider.print_join_plots(train_dir + 'cartoons/', 'images/joinplot_cartoons', 'C')
UtilProvider.print_join_plots(train_dir + 'photos/', 'images/joinplot_photos', 'P')

[train_generator, validation_generator] = UtilProvider.create_data_generators(batch_size, image_size)

# Set up TensorBoard
shutil.rmtree('logs')  # Clean existing logs directory before starting
log_dir = 'logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = run_tensorboard(log_dir)

# Compile and fit model
model = create_model()
history = fit_model()
print(model.history.history)

# Save model
UtilProvider.save_model(model, './models/simple_classifier.h5')

# Display metrics and classification report
UtilProvider.display_metrics(model, './images/s-loss-val_loss', './images/s-accuracy-val_accuracy')
UtilProvider.display_classification_report(model, validation_generator)

# Test predictions
UtilProvider.make_predictions(model)
