import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from PIL import Image
import matplotlib.pyplot as plt

def load_images_and_labels(directory, labels):
    X = []
    y = []
    
    for label in labels:
        label_dir = os.path.join(directory, label)
        files = os.listdir(label_dir)
        total_files = len(files)
        print(f"Processing {total_files} images in the {label} category...")
        
        for file_name in files:
            file_path = os.path.join(label_dir, file_name)
            img = Image.open(file_path).convert("L")
            img_array = np.array(img)
            X.append(img_array)
            y.append(labels.index(label))
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def preprocess_images(X_train, X_test, X_valid):
    # Normalize the pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.0

    # Reshape the image arrays
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    X_valid = X_valid.reshape(-1, 48, 48, 1)
    
    return X_train, X_test, X_valid

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid))
    return history

def save_model(model, model_file):
    model.save(model_file)

def generate_classification_report(model, X_test, y_test, labels, report_file):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    report = classification_report(y_test, y_pred_labels, target_names=labels)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Classification report saved successfully at {report_file}.")
    print(report)

def plot_training_history(history, epochs):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(range(epochs), history.history["loss"], label="train_loss")
    plt.plot(range(epochs), history.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(range(epochs), history.history["accuracy"], label="train_acc")
    plt.plot(range(epochs), history.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

def save_plot(plot_file):
    plt.savefig(plot_file)

# Set the root directory of the facial expressions dataset
root_dir = os.path.join("images")

# Define the list of facial expressions (class labels)
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Define number of classes, i.e. facial expressions for building model later
num_classes = len(labels)

# Load TRAINING images and labels
train_dir = os.path.join("..", root_dir, "train")
X_train, y_train = load_images_and_labels(train_dir, labels)
print("Training images loading complete.")

# Load VALIDATION images and labels
valid_dir = os.path.join("..", root_dir, "valid")
X_valid, y_valid = load_images_and_labels(valid_dir, labels)
print("Validation images loading complete.")

# Load TESTING images and labels
test_dir = os.path.join("..", root_dir, "test")
X_test, y_test = load_images_and_labels(test_dir, labels)
print("Test images loading complete.")

# Preprocess the images
X_train, X_test, X_valid = preprocess_images(X_train, X_test, X_valid)

# Build the model
model = build_model(num_classes)

# Train the model and store the history
epochs = 20
batch_size = 32
history = train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size)

# Save the model
model_file = "models/CNN.h5"
save_model(model, model_file)

# Generate and save the classification report
reports_dir = os.path.join("..", "reports")
os.makedirs(reports_dir, exist_ok=True)
report_file = os.path.join(reports_dir, "CNN.txt")
generate_classification_report(model, X_test, y_test, labels, report_file)

# Plot and save the training history
plots_folder = os.path.join("..", "plots")
os.makedirs(plots_folder, exist_ok=True)
plot_file = os.path.join(plots_folder, "CNN.png")
plot_training_history(history, epochs)
save_plot(plot_file)