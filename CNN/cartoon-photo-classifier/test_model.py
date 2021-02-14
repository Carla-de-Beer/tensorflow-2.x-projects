import tensorflow as tf

from util_provider import UtilProvider


def test_model(model_path):
    model = tf.keras.models.load_model(model_path)
    UtilProvider.make_predictions(model)


test_model('./models/simple_classifier.h5')
test_model('./models/transfer_classifier.h5')
