# Carla de Beer
# Created: March 2020
# A Keras-based deep learning autoencoder that filters out noise from a given image.
# The project uses the Fashion MNIST dataset:
# https://github.com/zalandoresearch/fashion-mnist2

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

epochs = 10


def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


def scale_data(X_train, X_test):
    X_train = X_train / 255
    X_test = X_test / 255
    return X_train, X_test


def create_encoder():
    encoder = Sequential()

    encoder.add(Flatten(input_shape=[28, 28]))
    encoder.add(Dense(400, activation='relu'))
    encoder.add(Dense(200, activation='relu'))
    encoder.add(Dense(100, activation='relu'))
    encoder.add(Dense(50, activation='relu'))
    encoder.add(Dense(25, activation='relu'))

    return encoder


def create_decoder():
    decoder = Sequential()

    decoder.add(Dense(50, input_shape=[25], activation='relu'))
    decoder.add(Dense(100, activation='relu'))
    decoder.add(Dense(200, activation='relu'))
    decoder.add(Dense(400, activation='relu'))
    decoder.add(Dense(784, activation='sigmoid'))

    decoder.add(Reshape([28, 28]))

    return decoder


def create_noise_remover(encoder, decoder):
    noise_remover = Sequential([encoder, decoder])
    noise_remover.compile(loss='binary_crossentropy', optimizer=SGD(lr=1.5), metrics=['accuracy'])
    noise_remover.fit(X_train, X_train, epochs=epochs, validation_data=[X_test, X_test])

    return noise_remover


def print_results():
    for n in range(10):
        num = str(n + 1)
        print('Processing Image: ' + num)

        plt.imshow(X_test[n])
        plt.savefig('images/original-image-' + num)
        plt.show()

        plt.imshow(ten_noisy_images[n])
        plt.savefig('images/noisy-image-' + num)
        plt.show()

        plt.imshow(denoised[n])
        plt.savefig('images/cleaned-image-' + num)
        plt.show()


(X_train, y_train), (X_test, y_test) = load_data()
print(X_train.shape)
plt.imshow(X_train[0])

X_train, X_test = scale_data(X_train, X_test)

# Build stacked autoencoder
print(28 * 28)
print((28 * 28) / 2)
print(25 / 784)  # => going down to 3% of the original features

encoder = create_encoder()
decoder = create_decoder()

noise_remover = create_noise_remover(encoder, decoder)

sample = GaussianNoise(0.4)
ten_noisy_images = sample(X_test[:10], training=True)

denoised = noise_remover(ten_noisy_images)

print_results()
