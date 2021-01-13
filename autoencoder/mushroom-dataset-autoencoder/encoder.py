import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler


def create_encoder():
    _encoder = Sequential()
    _encoder.add(Dense(units=12, activation='relu', input_shape=[21]))
    _encoder.add(Dense(units=2, activation='relu', input_shape=[12]))
    return _encoder


def create_decoder():
    _decoder = Sequential()
    _decoder.add(Dense(units=12, activation='relu', input_shape=[2]))
    _decoder.add(Dense(units=21, activation='relu', input_shape=[12]))
    return _decoder


def convert_chars_to_ints(_df):
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
    mapping_class = {'p': 1, 'e': 2}

    _df = _df.replace({'cap-shape': mapping_cap_shape})
    _df = _df.replace({'cap-surface': mapping_cap_surface})
    _df = _df.replace({'cap-color': mapping_cap_color})
    _df = _df.replace({'bruises': mapping_bruises})
    _df = _df.replace({'odor': mapping_odor})
    _df = _df.replace({'gill-attachment': mapping_gill_attachment})
    _df = _df.replace({'gill-spacing': mapping_gill_spacing})
    _df = _df.replace({'gill-size': mapping_gill_size})
    _df = _df.replace({'gill-color': mapping_gill_color})
    _df = _df.replace({'stalk-shape': mapping_stalk_shape})
    _df = _df.replace({'stalk-root': mapping_stalk_root})
    _df = _df.replace({'stalk-surface-above-ring': mapping_stalk_surface_above_ring})
    _df = _df.replace({'stalk-surface-below-ring': mapping_stalk_surface_below_ring})
    _df = _df.replace({'stalk-color-above-ring': mapping_stalk_color_above_ring})
    _df = _df.replace({'stalk-color-below-ring': mapping_stalk_color_below_ring})
    _df = _df.replace({'veil-type': mapping_veil_type})
    _df = _df.replace({'veil-color': mapping_veil_color})
    _df = _df.replace({'ring-number': mapping_ring_number})
    _df = _df.replace({'ring-type': mapping_ring_type})
    _df = _df.replace({'spore-print-color': mapping_spore_print_color})
    _df = _df.replace({'class': mapping_class})
    _df = _df.replace({'population': mapping_population})
    _df = _df.replace({'habitat': mapping_habitat})

    return _df


def plot_encoded_dims(data):
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(data[:, 0], data[:, 1], c=y)

    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.add_artist(legend)
    plt.savefig('images/2D_visualisation')
    plt.show()


df = pd.read_csv('data/mushrooms.csv')

print(df.head())
print(df.head())
print(df.shape)

labels = df['class']
df = convert_chars_to_ints(df)
df = df.drop(['stalk-root'], axis=1)

X = df.drop('class', axis=1)
X = X.to_numpy()
y = df['class']

feat = pd.DataFrame(X)
feat.head()

print(feat.shape)

encoder = create_encoder()
decoder = create_decoder()

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss='mse', optimizer=SGD(lr=1.0))

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(feat)

autoencoder.fit(scaled_data, scaled_data, epochs=20)

encoded_2dim = encoder.predict(scaled_data)

encoded_df = pd.DataFrame(encoded_2dim, columns=['X1', 'X2'])
encoded_df['classes'] = labels

plot_encoded_dims(encoded_2dim)
