import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


class UtilProvider:

    @staticmethod
    def create_data_generators(batch_size, image_size):
        # Define the example directories and files
        base_dir = 'data'

        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')

        # Add the data-augmentation parameters to ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        # Flow training images in batches of batch_size using train_datagen generator
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            shuffle=True,
                                                            target_size=(image_size, image_size))

        # Flow validation images in batches of batch_size using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                shuffle=False,
                                                                target_size=(image_size, image_size))

        return train_generator, validation_generator

    @staticmethod
    def save_model(model, model_path):
        model_path = model_path
        model.save(model_path)
        print(f"Model file size in bytes: {os.stat(model_path).st_size}")

    @staticmethod
    def display_classification_report(model, validation_generator):
        predictions = (model.predict(validation_generator) > 0.5).astype('int32')

        print(classification_report(validation_generator.labels, predictions))
        print(confusion_matrix(validation_generator.labels, predictions))

    @staticmethod
    def display_metrics(model, file_name_1, file_name_2):
        metrics = pd.DataFrame(model.history.history)

        metrics[['loss', 'val_loss']].plot(color=['turquoise', 'tomato'])

        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.title('Loss and Validation Loss', fontsize=10, fontweight='bold')

        plt.savefig(file_name_1)
        plt.show()

        metrics[['accuracy', 'val_accuracy']].plot(color=['turquoise', 'tomato'])

        plt.xlabel('Epochs')
        plt.ylabel('Percentage')
        plt.title('Accuracy and Validation Accuracy', fontsize=10, fontweight='bold')

        plt.savefig(file_name_2)
        plt.show()

    @staticmethod
    def print_join_plots(dir_name, output_name, type):
        dim1 = []
        dim2 = []

        if type == 'C':
            text = 'Cartoon Image Sizes'
        else:
            text = 'Photo Image Sizes'

        for image_filename in os.listdir(dir_name):
            if image_filename != '.DS_Store':
                img = imread(dir_name + image_filename)
                d1, d2, colors = img.shape
                dim1.append(d1)
                dim2.append(d2)

        print(len(dim1))

        df = pd.DataFrame()

        df['Height'] = pd.Series(dim1)
        df['Width'] = pd.Series(dim2)

        j_plot = sns.jointplot(data=df, x='Width', y='Height', kind='hist')
        j_plot.fig.suptitle(text)
        j_plot.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
        plt.savefig(output_name)
        plt.show()

    @staticmethod
    def make_predictions(model):
        dir_name = './unseen'

        unseen_dir = os.path.join('unseen')
        unseen_names = os.listdir(unseen_dir)
        unseen_names.remove('.DS_Store')  # In case of a macOS directory
        unseen_names.sort()

        count = 1
        error = 0.0

        for filename in unseen_names:
            if filename.endswith('.jpg'):
                with open(os.path.join(dir_name + '/', filename)) as f:
                    img = image.load_img(dir_name + '/' + filename, target_size=(256, 256))

                    my_img_array = image.img_to_array(img)
                    my_img_array = my_img_array / 255
                    my_img_array = np.expand_dims(my_img_array, axis=0)

                    classes = model.predict(my_img_array)

                    if classes[0][0] < 0.5:
                        print(str(count) + '. ' + filename + ': CARTOON')
                        print(f"Degree of certainty: {round(100 - classes[0][0] * 100, 4)}%")
                        error = error + pow(0 - classes[0][0], 2)
                    elif classes[0][0] >= 0.5:
                        print(str(count) + '. ' + filename + ': PHOTO')
                        print(f"Degree of certainty: {round(classes[0][0] * 100, 4)}%")
                        error = error + pow(1 - classes[0][0], 2)

                    count = count + 1

        print(f"Test error: {round(error / ((count - 1) * 2), 6) * 100}%")
        print('\nDONE.')
