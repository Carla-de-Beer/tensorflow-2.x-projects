# Image Filter Fashion MNIST

A Keras-based deep learning autoencoder that filters out noise from a given image.
The Fashion MNIST dataset is read in, fed into an encoder, which, in turn, is linked with a decoder.
A set of manually manipulated images with random noise added to them is then fed the autoencoder and a cleaner version of these images is then generated.

## Resources

* The project uses the Fashion MNIST dataset: https://github.com/zalandoresearch/fashion-mnist2.

### Images Generated

<table>
  <tr>
    <th>Original Image</th>
    <th>Noisy Image</th>
    <th>Filtered Image</th>
  </tr>
  
  <tr>
    <th>
      <img src="images/original-image-1.png" width="250px"/>
    </th>
    <th>
      <img src="images/noisy-image-1.png" width="250px"/>
    </th>
    <th>  
      <img src="images/cleaned-image-1.png" width="250px"/>
    </th>
  </tr>
  <tr>
    <th>
      <img src="images/original-image-3.png" width="250px"/>
    </th>
    <th>
      <img src="images/noisy-image-3.png" width="250px"/>
    </th>
    <th>
      <img src="images/cleaned-image-3.png" width="250px"/>
    </th>
  </tr>
  <tr>
    <th>
      <img src="images/original-image-5.png" width="250px"/>
    </th>
    <th>
      <img src="images/noisy-image-5.png" width="250px"/>
    </th>
    <th>
      <img src="images/cleaned-image-5.png" width="250px"/>
    </th>
  </tr>
  <tr>
    <th>
      <img src="images/original-image-7.png" width="250px"/>
    </th>
    <th>
      <img src="images/noisy-image-7.png" width="250px"/>
    </th>
    <th>
      <img src="images/cleaned-image-7.png" width="250px"/>
    </th>
  </tr>
</table>
<br/>
