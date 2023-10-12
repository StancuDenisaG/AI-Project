#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
from skimage import io, transform
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#functie ce genereaza imagini in batch-uri
def image_generator(image_paths, batch_size):
    # calculam nr de batch-uri
    num_batches = len(image_paths) // batch_size
    for i in range(num_batches):
        batch_paths = image_paths[i*batch_size:(i+1)*batch_size]
        # transformam imaginile
        batch_images = [transform.resize(io.imread(image), (64, 64)).astype(np.float32) for image in batch_paths]
        # transformam in NumPy array 
        yield np.array(batch_images).reshape(batch_size, -1)
    # luam restul imaginilor ramase
    remaining_paths = image_paths[num_batches*batch_size:]
    if len(remaining_paths) > 0:
        remaining_images = [transform.resize(io.imread(image), (64, 64)).astype(np.float32) for image in remaining_paths]
        yield np.array(remaining_images).reshape(len(remaining_paths), -1)

#citim si luam datele de antrenare si validare
df_train = pd.read_csv("train_labels.txt", dtype={'id': 'string', 'class': 'int32'})
df_train['id'] = 'data/' + df_train['id'] + '.png'
trainPaths = list(df_train['id'])
trainLabels = list(df_train['class'])


df_val = pd.read_csv("validation_labels.txt", dtype={'id': 'string', 'class': 'int32'})
df_val['id'] = 'data/' + df_val['id'] + '.png'
validationPaths = list(df_val['id'])
validationLabels = list(df_val['class'])

#procesam datele prin normalizarea valorilor de pixeli si convertirea etichetelor în format binar
trainData = np.array(list(image_generator(trainPaths, len(trainPaths))))
trainData = trainData.reshape(len(trainPaths), 64, 64, 3)
trainData = (trainData - np.mean(trainData)) / np.std(trainData)
trainLabels = to_categorical(trainLabels)

validationData = np.array(list(image_generator(validationPaths, len(validationPaths))))
validationData = validationData.reshape(len(validationPaths), 64, 64, 3)
validationData = (validationData - np.mean(validationData)) / np.std(validationData)
validationLabels = to_categorical(validationLabels)


# In[2]:


# definim modelul
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')
    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainData, trainLabels, validation_data=(validationData, validationLabels), epochs=6, batch_size=100)


_, acc = model.evaluate(validationData, validationLabels, verbose=0)
print(f"Validation accuracy: {acc*100:.2f}%")


# In[3]:


#graf pentru reprezentarea acuratetei pe datele de validare
train_accuracy = history.history['accuracy']
validationAccuracy = history.history['val_accuracy']
epochs = range(1, 7)

plt.plot(epochs, validationAccuracy, 'b', label='Accuracy on validation data')
plt.title('Accuracy on Validation Data')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

validationLoss, validationAccuracy = model.evaluate(validationData, validationLabels, verbose=2)
predictedValidationLabels = model.predict(validationData)     # prezicem etichetele pe datele de validare pt matricea de confuzie
predictedValidationLabels = np.argmax(np.round(predictedValidationLabels), axis=1)   

print("Accuracy on validation data: ", validationAccuracy)

#matricea de confuzie
confMatrix = confusion_matrix(validationLabels.argmax(axis=1), predictedValidationLabels, labels=[0,1])
display = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=[0,1]) 
display.plot()
plt.show()


# In[5]:


df_test = pd.read_csv("sample_submission.txt", dtype={'id': 'string'})
df_test['id'] = 'data/' + df_test['id'] + '.png'
testPaths = list(df_test['id'])
testData = np.array(list(image_generator(testPaths, len(testPaths))))
testData = testData.reshape(len(testPaths), 64, 64, 3)
testData = (testData - np.mean(testData)) / np.std(testData)




# prezicem etichetele pe imaginile de testare
predLabels = model.predict(testData)


# scriem etichetele in fisier 
with open("predictionsCNN.txt" , 'w') as f:
    f.write("id,class\n")
    for i in range(len(predLabels)):
        predClass = np.argmax(predLabels[i])
        f.write(list(df_test['id'])[i].replace('data/', '').replace('.png', '') + ',' + str(predClass) + '\n')






