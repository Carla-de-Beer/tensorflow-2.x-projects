import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping

from data_handler import DataHandler
from util_provider import UtilProvider

local_weights_file = './models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

image_size = 256
image_shape = (image_size, image_size, 3)
epochs = 20
patience = 3
batch_size = 4


def get_pre_trained_model():
    pre_trained_model = InceptionV3(input_shape=(image_size, image_size, 3),
                                    include_top=False,
                                    weights=None)

    DataHandler.extract_inception_model(local_weights_file)
    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    print(pre_trained_model.summary())

    return pre_trained_model


def create_compile_model(pre_trained_model):
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension and
    # add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def fit_model():
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)

    return model.fit(train_generator,
                     validation_data=validation_generator,
                     steps_per_epoch=train_steps_per_epoch,
                     epochs=epochs,
                     validation_steps=val_steps_per_epoch,
                     verbose=2,
                     callbacks=[early_stop])


# Run code

# Get and process data
DataHandler.extract_data_files()

num_images = DataHandler.calculate_num_images()
train_steps_per_epoch = np.ceil((num_images * 0.8 / batch_size) - 1)
val_steps_per_epoch = np.ceil((num_images * 0.2 / batch_size) - 1)

[train_generator, validation_generator] = UtilProvider.create_data_generators(batch_size, image_size)

# Compile and fit model
model = create_compile_model(get_pre_trained_model())
print(model.summary())
history = fit_model()
print(model.history.history)

# Save model
UtilProvider.save_model(model, './models/transfer_classifier.h5')

# Display metrics and classification report
UtilProvider.display_metrics(model, './images/t-loss-val_loss', './images/t-accuracy-val_accuracy')
UtilProvider.display_classification_report(model, validation_generator)

# Test predictions
UtilProvider.make_predictions(model)
