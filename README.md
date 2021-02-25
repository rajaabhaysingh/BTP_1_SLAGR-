# COMMUNICATION SYSTEM FOR DEAF, DUMB AND BLIND(USING GESTURE RECOGNITION)

Mobile technology is very fast growing and incredible, yet there are not many technological development and improvement for deaf and mute peoples. We are aiming to come up with a reliable platform to channelize communication between normal and deaf/mute person and vice versa.

To provide a communication channel to specially-abled people, specifically – deaf, mute and blind, using “gesture tracking and recognition” with the help of computer vision and deep learning.

In this project, We will use Indo-Pakistani Sign Language (IPSL) to train and test our primitive model. It will only be able to recognize hand gesture.

We investigate different machine learning techniques like:
- [K-Nearest-Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## Getting Started
### Prerequisites
Before running this project, make sure you have following dependencies - 
* [Dataset](https://drive.google.com/file/d/15bikHgG8Y13vWdMQ-6-MK0y8AI3sGBfe/view?usp=sharing) (Download the images from this link)
* [Python 3.6](https://www.python.org/downloads/)
* [pip](https://pypi.python.org/pypi/pip)
* [OpenCV](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)

Now, using ```pip install``` command, include following dependencies 
+ Numpy 
+ Pandas
+ Sklearn
+ Scipy
+ Opencv
+ Tensorflow

### Running
To run the project, perform following steps -

 1. Put all the training and testing images in a directory and update their paths in the config file *`common/config.py`*.
 2. Generate image-vs-label mapping for all the training images - `generate_images_labels.py train`.
 3. Apply the image-transformation algorithms to the training images - `transform_images.py`.
 4. Train the model(KNN & SVM) - `train_model.py <model-name>`. Note that the repo already includes pre-trained models for some algorithms serialized at *`data/generated/output/<model-name>/model-serialized-<model-name>.pkl`*.
 5. Generate image-vs-label mapping for all the test images - `generate_images_labels.py test`.
 6. Test the model - `predict_from_file.py <model-name>`.
  

