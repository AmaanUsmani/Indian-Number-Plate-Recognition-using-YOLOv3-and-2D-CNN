import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the labeled images and labels
labeled_images = []
labels = []
for filename in os.listdir('labeled_images'):
    if filename.endswith('.png') and '_' in filename:
        img = cv2.imread(os.path.join('labeled_images', filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        label_str = filename.split('_')[-1].split('.')[0]
        try:
            label = int(label_str)
        except ValueError:
            label = label_str
        labeled_images.append(img)
        labels.append(label)

# Convert the label strings to integers using a label encoder
le = LabelEncoder()
labels = le.fit_transform(labels)

# Resize and preprocess the images
image_size = (48, 48)
num_channels = 3

labeled_images_resized = []
for img in labeled_images:
    img_resized = cv2.resize(img, image_size)
    img_resized = img_resized.astype('float32') / 255.0
    labeled_images_resized.append(img_resized)

# Split the labeled images and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(labeled_images_resized, labels, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoded vectors
num_classes = len(le.classes_)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], num_channels))

x = Flatten()(base_model.output)
output_layer = Dense(num_classes, activation='softmax')(x)

# Freeze all layers except for the new output layer
for layer in base_model.layers:
    layer.trainable = False
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), validation_split=0.2, epochs=20, batch_size=32)

# Evaluate the performance of the trained model
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test accuracy: {accuracy:.2f}%')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
total_count = len(X_test)
accuracy = 0
y_pred=[]

# Resize and preprocess the images
image_size = (48, 48)
num_channels = 3

labeled_images_resized = []
for img in X_test:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img, image_size)
    img_resized = img_resized.astype('float32') / 255.0
    labeled_images_resized.append(img_resized)

X_test = np.array(labeled_images_resized)

# Create a lookup table for converting labels to characters
label_table = {}
for i in range(10):
    label_table[i] = str(i)
for i in range(26):
    label_table[i+10] = chr(i+97)

for i in range(total_count):
    true_label = label_table[y_test[i]]

    # Preprocess the image
    img = X_test[i]
    img = np.expand_dims(img, axis=0)
    # Use the trained model to predict the label
    pred = model.predict(img)
    predicted_label = label_table[np.argmax(pred)]
    print(predicted_label)
    print(true_label)
    y_pred.append(predicted_label.lower())
    if predicted_label.lower() == true_label.lower():
        accuracy += 1
        
accuracy_percent = accuracy / total_count * 100 if total_count > 0 else 0

print(f"Accuracy: {accuracy_percent:.2f}%")
print(f"Total Count: {total_count}")

# Convert true labels to numeric form
y_true_numeric = [int(y_test[i]) for i in range(total_count)]

# Convert predicted labels to numeric form
y_pred_numeric = []
for l in y_pred:
    if l in label_table.values():
        y_pred_numeric.append(list(label_table.keys())[list(label_table.values()).index(l)])
    else:
        y_pred_numeric.append(-1) # -1 indicates that the label is not in the label_table dictionary

# Calculate and plot confusion matrix
cm = confusion_matrix(y_true_numeric, y_pred_numeric,  labels=list(range(36)))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', xticklabels=label_table.values(), yticklabels=label_table.values())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Generate classification report
class_names = [label_table[i] for i in range(34)]
report = classification_report(y_true_numeric, y_pred_numeric, target_names=class_names)

print(report)