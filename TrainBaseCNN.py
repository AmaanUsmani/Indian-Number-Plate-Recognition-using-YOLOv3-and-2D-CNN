import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import model_to_dot
import pydotplus
from IPython.display import Image
# from cv2 import dnn

import util


char_images = []
# for filename in os.listdir('char_images'):
#     img = cv2.imread(os.path.join('char_images', filename), cv2.IMREAD_GRAYSCALE)
#     img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     img = cv2.resize(img, (48, 48))  # Resize the image to a fixed size
#     char_images.append(img)

# # Convert the list of images to a 2D array for K-means clustering
# char_array = np.array(char_images).reshape(len(char_images), -1)

# # Cluster the images using K-means clustering
# kmeans = KMeans(n_clusters=36, random_state=42).fit(char_array)

# # Save the labeled images to the labeled_images directory
# if not os.path.exists('labeled_images'):
#     os.mkdir('labeled_images')
# for i, label in enumerate(kmeans.labels_):
#     cv2.imwrite(f'labeled_images/char_{i}_{label}.png', char_images[i])



# Load the labeled images and labels
labeled_images = []
labels = []
for filename in os.listdir('labeled_images'):
    if filename.endswith('.png') and '_' in filename:
        img = cv2.imread(os.path.join('labeled_images', filename), cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 64 , 255, cv2.THRESH_BINARY_INV)
        img = np.array(img).reshape(48, 48, 1)
        label_str = filename.split('_')[-1].split('.')[0]
        try:
            label = int(label_str)
        except ValueError:
            label = label_str
        labeled_images.append(img)
        labels.append(label)

labeled_images = np.array(labeled_images)
print(labeled_images.shape)

# Split the labeled images and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(labeled_images, labels, test_size=0.2, random_state=42)

# X_train[0].shape
# Convert the label strings to integers using a label encoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Create the neural network model
model = keras.Sequential(
    [
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),    
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(le.classes_), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
plot_model(model, show_shapes=True, show_layer_names=True)
plt.show()
# Train the neural network model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))



# Evaluate the neural network model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Print the incorrectly predicted labels and keep count of incorrect predictions per label
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
incorrect_labels = []
incorrect_counts = {}
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        incorrect_labels.append((le.inverse_transform([y_test[i]])[0], le.inverse_transform([y_pred[i]])[0]))
        true_label = le.inverse_transform([y_test[i]])[0]
        pred_label = le.inverse_transform([y_pred[i]])[0]
        if true_label not in incorrect_counts:
            incorrect_counts[true_label] = {}
        if pred_label not in incorrect_counts[true_label]:
            incorrect_counts[true_label][pred_label] = 1
        else:
            incorrect_counts[true_label][pred_label] += 1

if incorrect_labels:
    print(f"Incorrectly predicted labels: {incorrect_labels}")
    print("Incorrect predictions count per label:")
    for true_label in incorrect_counts:
        for pred_label in incorrect_counts[true_label]:
            count = incorrect_counts[true_label][pred_label]
            print(f"True label: {true_label}, Predicted label: {pred_label}, Count: {count}")
else:
    print("All labels predicted correctly")

# Plot the training and validation accuracy over the epochs
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# np.save("X_test.npy", X_test)
# np.save("y_test.npy", y_test)
# model.save('model3_without_feature_extraction.h5')