import easyocr
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

reader = easyocr.Reader(['en'])

# Create a lookup table for converting labels to characters
label_table = {}
for i in range(10):
    label_table[i] = str(i)
for i in range(26):
    label_table[i+10] = chr(i+97)

x_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

image_folder = 'labeled_images'
total_count = len(x_test)
accuracy = 0
y_pred=[]

for i in range(total_count):
    true_label = label_table[y_test[i]]
    img = x_test[i]
    
    # Convert image to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize image to 64x64 pixels
    img = cv2.resize(img, (48,48))
    _, img = cv2.threshold(img, 0 , 255, cv2.THRESH_BINARY)
    # Use EasyOCR to predict label
    result = reader.readtext(img)
    if result and result[0][1].isalnum() and len(result[0][1]) == 1:
        predicted_label = result[0][1]
    else:
        predicted_label = random.choice(list(label_table.values()))

    y_pred.append(predicted_label.lower())

    print(f"True label: {true_label}, Predicted label: {predicted_label}")

    if predicted_label and predicted_label[0].lower() == true_label.lower():
        print('Match found')
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
        
# Get a list of all the characters in the label table
all_chars = set(label_table.values())

# Get a list of unique characters in y_true_numeric
unique_chars = set(y_true_numeric)

# Get a list of characters that do not exist in y_true_numeric
missing_chars = list(all_chars - unique_chars)

print(f"Missing characters: {missing_chars}")

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred_numeric, labels=list(range(36)))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', xticklabels=label_table.values(), yticklabels=label_table.values())
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Generate classification report
class_names = [label_table[i] for i in range(36)]
report = classification_report(y_true_numeric, y_pred_numeric, target_names=class_names)

print(report)