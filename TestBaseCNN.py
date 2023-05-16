import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Create a lookup table for converting labels to characters
label_table = {}
for i in range(10):
    label_table[i] = str(i)
for i in range(26):
    label_table[i+10] = chr(i+65)

# Load the trained model
model = load_model('mymodel.h5')

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

total_count = len(X_test)
accuracy = 0
y_pred = []

for i in range(total_count):
    true_label = label_table[y_test[i]]

    # Preprocess the image
    img = X_test[i].astype(np.uint8)
    # img = cv2.resize(img, (48, 48))
    # _, img = cv2.threshold(img, 64 , 255, cv2.THRESH_BINARY_INV)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.expand_dims(img, axis=0)
    # Use the trained model to predict the label
    pred = model.predict(img)
    predicted_label = label_table[np.argmax(pred)]
    y_pred.append(np.argmax(pred))

    if predicted_label.lower() == true_label.lower():
        accuracy += 1
        
accuracy_percent = accuracy / total_count * 100 if total_count > 0 else 0

print(f"Accuracy: {accuracy_percent:.2f}%")
print(f"Total Count: {total_count}")

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(34)))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', xticklabels=label_table.values(), yticklabels=label_table.values())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Generate classification report
class_names = [label_table[i] for i in range(34)]
report = classification_report(y_test, y_pred, target_names=class_names)

print(report)