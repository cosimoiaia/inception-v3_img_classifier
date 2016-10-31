# inception-v3 image classifier
Simple parametized python script to use a fine trained Inception V3 model to classify images.

Based on:
*  Tensorflow example https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html#training-on-your-own-categories
*  This great article on codeLabs https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

NOTE: This version will work only with TensorFlow-0.9.0-devel!

<b>Dependencies</b>
* Python >= 2.7
* Tensorflow 0.9.0-devel (see https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)

<b>Usage:</b>
* Fine train the Inception v3 model using the train.sh script:
   ```bash
   $ ./train.sh --tf_bin=/path/to/tensorflow/installation --tf_data=/path/to/images/data/folder.
   ```
   You can put any number of sub-directories in your data folder, inception will be fine trained to classify
   any images in categories define by those sub-subdirectories.
   ```
   i.e.
   /cat
    -> Persian
    -> Bengal
    -> Burmese
    -> Ragdoll
   ```
   Will train inception to classify any picture into those 4 Cat's breeeds cathegories.

* Classify your images with label_image.py:
   ```bash
   $ ./label_image.py --datafolder=/tensorflow --image_path=img/cat.jpg
   Persian (score = 0.88331)
   Bengal (score = 0.11669)
   Burmese (score = 0.23879)
   Ragdoll (score = 0.17469)
   ```
