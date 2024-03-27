import numpy as np
import os
from PIL import Image
import tkinter as tk
#GUI
from tkinter import filedialog

# Set the image size and channels
CHANNELS = 3
IMG_SIZE = 50

# Define the image paths and labels
CAT_PATH = "Cat/"
HUMAN_PATH = "Human/"
CAT_LABEL = 0
HUMAN_LABEL = 1

# Read and preprocess the image data
data = []
labels = []

# Read the cat images
for file in os.listdir(CAT_PATH):
    if file.endswith(".jpg"or".jpeg"):
        img_path = os.path.join(CAT_PATH, file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        data.append(np.array(img).flatten())
        labels.append(CAT_LABEL)

# Read the human images
for file in os.listdir(HUMAN_PATH):
    if file.endswith(".jpg"):
        img_path = os.path.join(HUMAN_PATH, file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        data.append(np.array(img).flatten())
        labels.append(HUMAN_LABEL)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize the data
data = data / 255.0

# Define the perceptron model
class Perceptron:
    def __init__(self, input_size, lr=0.01, epochs=100):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    # Return Y in the Equation
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias#NET
        a = self.activation_fn(z)
        return a

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(y.shape[0]):
                x = X[i]

                y_hat = self.predict(x)
                error = y[i] - y_hat #Error = T-Y
                # NewWeight = OldWeight +(delta *error * X)
                self.weights += self.lr * error * x
                self.bias += self.lr * error


# Create the perceptron model object and train it on the data
perceptron = Perceptron(input_size=data.shape[1])
perceptron.train(data, labels)

# Create a GUI for the model
root = tk.Tk()
root.title("Cat or Human?")
root.geometry("400x200")

# Define the function for predicting the image
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_data = np.array(img).flatten() / 255.0
        prediction = perceptron.predict(img_data)
        if prediction == CAT_LABEL:
            result_label.config(text="The image is of a cat.")
        else:
            result_label.config(text="The image is of a human.")
    else:
        result_label.config(text="Please choose an image.")

# Create a button for selecting an image
select_button = tk.Button(root, text="Select Image", command=predict_image)
select_button.pack(pady=20)

# Create a label for displaying the result

result_label = tk.Label(root, text="")
result_label.config(font=("Arial", 20, "bold"))

result_label.pack()

root.mainloop()
