import numpy as np
import os
from PIL import Image, ImageTk  # Import ImageTk for displaying images in tkinter
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set image size and channels
CHANNELS = 3
IMG_SIZE = 224  # Change the image size to 224x224

# Define data paths and labels
CAT_PATH = "Cat/"
HUMAN_PATH = "Human/"
CAT_LABEL = 0
HUMAN_LABEL = 1

# Function to load, preprocess, and return data and labels
def load_data(path, label):
    data = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img_path = os.path.join(path, file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize images to 224x224
            img_data = np.array(img) / 255.0  # Normalize
            data.append((img_data, img))  # Store both image data and PIL image object
    return np.array([item[0] for item in data]), np.full((len(data),), label), [item[1] for item in data]

# Load cat and human data
cat_data, cat_labels, cat_images = load_data(CAT_PATH, CAT_LABEL)
human_data, human_labels, human_images = load_data(HUMAN_PATH, HUMAN_LABEL)

# Combine data and labels
data = np.concatenate((cat_data, human_data), axis=0)
labels = np.concatenate((cat_labels, human_labels), axis=0)
images = cat_images + human_images  # Combine image lists

# Split data into training and validation sets
train_data, val_data, train_labels, val_labels, train_images, val_images = train_test_split(data, labels, images, test_size=0.2, random_state=42)

# Reshape data for CNN (add channel dimension)
train_data = train_data.reshape(train_data.shape[0], IMG_SIZE, IMG_SIZE, CHANNELS)
val_data = val_data.reshape(val_data.shape[0], IMG_SIZE, IMG_SIZE, CHANNELS)

# Define the CNN model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))
model.add(MaxPooling2D((2, 2)))

# Flatten the data for feeding to the fully connected layers
model.add(Flatten())

# Fully connected layer 1
model.add(Dense(64, activation='relu'))

# Output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

# Function to predict on a new image
def predict_image():
    file_path = filedialog.askopenfilename()  # Open file dialog to select image
    if file_path:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize image to 224x224
        img_data = np.expand_dims(np.array(img) / 255.0, axis=0)  # Add batch dimension
        prediction = model.predict(img_data)[0][0]
        if prediction > 0.5:  # Threshold for human probability
            result_label.config(text="The image is of a human.", fg="blue")
        else:
            result_label.config(text="The image is of a cat.", fg="green")
        # Convert PIL image to Tkinter PhotoImage and display it
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection

# Create the GUI
root = tk.Tk()
root.title("Cat or Human?")
root.geometry("600x400")  # Adjusted size for better display

# Add a frame for better organization
frame = tk.Frame(root, bg="white")
frame.pack(fill=tk.BOTH, expand=True)

# Label to display the result
result_label = tk.Label(frame, text="", font=("Arial", 20, "bold"))
result_label.pack(pady=20)

# Label to display the image
img_label = tk.Label(frame)
img_label.pack(pady=20)

# Button to select an image
select_button = tk.Button(frame, text="Select Image", command=predict_image)
select_button.pack(pady=20)

# Run the GUI main loop
root.mainloop()
