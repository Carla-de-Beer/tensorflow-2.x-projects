from pathlib import Path
from util_provider import UtilProvider
import tensorflow as tf


def test_model(model_path):
    my_file = Path(model_path)
    if my_file.is_file():
        model = tf.keras.models.load_model(model_path)
        UtilProvider.make_predictions(model)
    else:
        print("Could not open the model file " + model_path)


test_model('./models/simple_classifier.h5')
test_model('./models/transfer_classifier.h5')
