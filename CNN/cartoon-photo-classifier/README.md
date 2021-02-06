# Cartoon-Photo Classifier

Two comparative Tensorflow-Keras-based deeplearning convolutional neural networks (CNN) to predict whether an image is a
cartoon or a photo. The simpler CNN makes use solely of the provided dataset, whereas the other CNN additionally
benefits from the application of the weights from a pre-trained model in the process of transfer learning. The classifier
based on transfer learning boosts the accuracy of the somewhat limited dataset to nearly 100%, and additionally manages
to train more quickly in only a handful of epochs. That said, even in the case of the simpler classifier, the model managed to achieve a reasonable degree of accuracy on a relatively small dataset.

## Dataset

Cartoons come in varying genres and styles. They are characterised mostly by a non-realistic or semi-realistic style,
with vibrant or garish colours in a saturated colour palette and often with outline contours to the figures depicted.
The images used in this dataset tend to conform to this paradigm. Likewise, the photos were chosen to reflect animals,
faces and landscapes similar to those featured in the cartoons. The aim of the CNNs is to see whether the neural
networks are able to find and learn the distinguishing features between the two types of images.

There are 250 JPEG images of either category, of varying sizes, and with an 8:2 split between training and validation
data. Images were sourced by means of Google searches.

### Image Size Distributions

<br/>
<p align="center">
  <img src="images/joinplot_cartoons.png" width="400px" alt="joinplot_cartoons"/>
  <img src="images/joinplot_photos.png" width="400px" alt="joinplot_photos"/>
</p>

## 1. Simple Classifier

The original dataset of cartoons and images, with a combined total of 500 images, is quite small in comparison to
typical CNN datasets, and also due to the many different subsets of styles and subjects featured in the cartoons and
photos. This may lead to some degree of overfitting, hence the addition of dropout and regularisation to this CNN. The
sourcing of more images should get around this issue and increase accuracy.

The simple classifier project is also mapped via TensorBoard. To follow the image and graph data there, do the
following:

* Once the code is running, run this command inside the terminal:
  ```tensorboard --logdir logs```.

* Open a browser window with the following URL:
  ```http://localhost:6006/```
  
#### TensorBoard Images

<br/>
<p align="center">
  <img src="images/tensorboard-01.png" width="650px" alt="tensorboard-01"/>
</p>

#### TensorBoard Graphs

<br/>
<p align="center">
  <img src="images/tensorboard-02.png" width="650px" alt="tensorboard-02"/>
</p>

### Model Evaluation

#### Accuracy and Losses During Training

<br/>
<p align="center">
  <img src="images/s-accuracy-val_accuracy.png" width="450px" alt="accuracy"/>
  <img src="images/s-loss-val_loss.png" width="450px" alt="loss"/>
</p>

#### Classification Report

```
              precision    recall  f1-score   support

           0       0.64      0.98      0.77        48
           1       0.95      0.44      0.60        48

    accuracy                           0.71        96
   macro avg       0.79      0.71      0.69        96
weighted avg       0.79      0.71      0.69        96
```

#### Confusion Matrix

```
[[47  1]
 [27 21]]
 ```

#### Testing with Unseen Data

| Filename      | Image                                                       | Prediction   | Result  |
| ------------- | -----------------------------------------------------------:| ------------:|--------:|
| image-01.jpg  |<img src="unseen/image-01.jpg" width="120px" alt="image-01"/>| CARTOON      | ✅      |
| image-02.jpg  |<img src="unseen/image-02.jpg" width="120px" alt="image-02"/>| CARTOON      | ✅      |
| image-03.jpg  |<img src="unseen/image-03.jpg" width="120px" alt="image-03"/>| CARTOON      | ✅      |
| image-04.jpg  |<img src="unseen/image-04.jpg" width="120px" alt="image-04"/>| CARTOON      | ✅      |
| image-05.jpg  |<img src="unseen/image-05.jpg" width="120px" alt="image-05"/>| CARTOON      | ✅      |
| image-06.jpg  |<img src="unseen/image-06.jpg" width="120px" alt="image-06"/>| CARTOON      | ❌      |
| image-07.jpg  |<img src="unseen/image-07.jpg" width="120px" alt="image-07"/>| PHOTO        | ✅      |
| image-08.jpg  |<img src="unseen/image-08.jpg" width="120px" alt="image-08"/>| PHOTO        | ✅      |
| image-09.jpg  |<img src="unseen/image-09.jpg" width="120px" alt="image-09"/>| CARTOON      | ❌      |
| image-10.jpg  |<img src="unseen/image-10.jpg" width="120px" alt="image-10"/>| PHOTO        | ✅      |

`image-06.jpg` and `image-09.jpg` are very colourful, as one would expect to see in a cartoon.
In one case, the model achieved an accuracy level of around 86%. In this case `image-09.jpg` was correctly classified.

## 2. Transfer Classifier

In an attempt to get around any overfitting of the first attempt, and to increase the model's accuracy without having 
to source additional dataset material, a second classifier is configured to make use of transfer learning. 
The transfer learning is based on the [Inception V3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3) model.

### Model Evaluation

#### Accuracy and Losses During Training

<br/>
<p align="center">
  <img src="images/t-accuracy-val_accuracy.png" width="450px" alt="accuracy"/>
  <img src="images/t-loss-val_loss.png" width="450px" alt="loss"/>
</p>

#### Classification Report

```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        48
           1       0.98      1.00      0.99        48

    accuracy                           0.99        96
   macro avg       0.99      0.99      0.99        96
weighted avg       0.99      0.99      0.99        96
```

#### Confusion Matrix

```
[[45  3]
 [ 0 48]]
 ```

#### Testing with Unseen Data

| Filename      | Image                                                       | Prediction   | Result  |
| ------------- | -----------------------------------------------------------:| ------------:|--------:|
| image-01.jpg  |<img src="unseen/image-01.jpg" width="120px" alt="image-01"/>| CARTOON      | ✅      |
| image-02.jpg  |<img src="unseen/image-02.jpg" width="120px" alt="image-02"/>| CARTOON      | ✅      |
| image-03.jpg  |<img src="unseen/image-03.jpg" width="120px" alt="image-03"/>| CARTOON      | ✅      |
| image-04.jpg  |<img src="unseen/image-04.jpg" width="120px" alt="image-04"/>| CARTOON      | ✅      |
| image-05.jpg  |<img src="unseen/image-05.jpg" width="120px" alt="image-05"/>| CARTOON      | ✅      |
| image-06.jpg  |<img src="unseen/image-06.jpg" width="120px" alt="image-06"/>| PHOTO        | ✅      |
| image-07.jpg  |<img src="unseen/image-07.jpg" width="120px" alt="image-07"/>| PHOTO        | ✅      |
| image-08.jpg  |<img src="unseen/image-08.jpg" width="120px" alt="image-08"/>| PHOTO        | ✅      |
| image-09.jpg  |<img src="unseen/image-09.jpg" width="120px" alt="image-09"/>| PHOTO        | ✅      |
| image-10.jpg  |<img src="unseen/image-10.jpg" width="120px" alt="image-10"/>| PHOTO        | ✅      |
