# Cat or Human Classification

This project implements a Convolutional Neural Network (CNN) model to classify images as either cats or humans. It utilizes the TensorFlow library with Keras API for building and training the CNN model, and tkinter for creating a simple GUI to interact with the model.This project is developed as part of a neural network course in college.
## Dependencies

The application requires the following Python libraries:

- Python 3.x
- TensorFlow
- PIL (Python Imaging Library)
- tkinter
- scikit-learn (for train-test split)
## How it Works

This Python script is a Convolutional Neural Network (CNN) based image classifier that distinguishes between images of cats and humans. It uses TensorFlow and Keras for building and training the model, and Tkinter for creating a graphical user interface (GUI). The script starts by importing necessary libraries and setting the image size and channels. It then defines paths to the directories containing the cat and human images and the labels for each class. A function `load_data` is defined to load and preprocess images from a given path, which includes resizing, normalizing pixel values, and returning the processed images along with their labels. The script then loads and preprocesses the cat and human images, combines the data and labels into single arrays, and splits the data into training and validation sets. A simple CNN model is defined with one convolutional layer, one max pooling layer, one fully connected layer, and an output layer for binary classification. The model is compiled with binary cross-entropy loss, Adam optimizer, and accuracy metric, and then trained on the training data for 10 epochs. The trained model is saved to a file named "mode.h5". A function `predict_image` is defined to open a file dialog for the user to select an image, preprocess the selected image, predict its class using the trained model, and display the prediction result and the image in the GUI. Finally, a simple GUI is created with a frame, a label to display the prediction result, a label to display the selected image, and a button to select an image, and the Tkinter main loop is run to start the GUI.

## Usage

To use the application, simply run the `main.py` script. A GUI will appear. Click the "Select Image" button to choose an image. The application will then predict whether the image is of a cat or a human and display the result.
