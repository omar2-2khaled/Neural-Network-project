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

The application reads all JPEG images from the specified directories, resizes them to the specified size, and converts them to numpy arrays. The arrays are flattened and normalized to a range of 0-1. Corresponding labels are created for each image.

The Perceptron model is a simple binary classification model. It is initialized with zeros for weights and bias. The learning rate and number of epochs are configurable. The model uses a step function as the activation function. The predict method calculates the dot product of the input and weights, adds the bias, and applies the activation function. The train method adjusts the weights and bias based on the error between the predicted and actual labels.

The application uses tkinter to create a simple GUI. The GUI includes a button for selecting an image and a label for displaying the prediction result. The predict_image function is called when the button is clicked. It opens a file dialog to select an image, preprocesses the image in the same way as the training images, and uses the Perceptron model to predict the label. The prediction result is displayed in the label.

## Usage

To use the application, simply run the `main.py` script. A GUI will appear. Click the "Select Image" button to choose an image. The application will then predict whether the image is of a cat or a human and display the result.
