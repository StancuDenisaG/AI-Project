#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


# In[2]:


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
    #luam restul imaginilor ramase
    remaining_paths = image_paths[num_batches*batch_size:]
    if len(remaining_paths) > 0:
        remaining_images = [transform.resize(io.imread(image), (64, 64)).astype(np.float32) for image in remaining_paths]
        yield np.array(remaining_images).reshape(len(remaining_paths), -1)


# In[3]:


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
trainData = trainData.reshape(len(trainPaths), -1)
trainData = (trainData - np.mean(trainData)) / np.std(trainData)

validationData = np.array(list(image_generator(validationPaths, len(validationPaths))))
validationData = validationData.reshape(len(validationPaths), -1)
validationData = (validationData - np.mean(validationData)) / np.std(validationData)


# In[4]:


# cream modelul SVM
model = svm.SVC(kernel='poly', C=3) 
batch_size = 1000
for i, batch_images in enumerate(image_generator(trainPaths, batch_size)):
    batch_images = (batch_images - np.mean(batch_images)) / np.std(batch_images)
    batch_labels = trainLabels[i*batch_size:(i+1)*batch_size]
    model.fit(batch_images, batch_labels)
    


# In[5]:


# prezicem etichetele pe datele de validare pt matricea de confuzie
predictedLabels = model.predict(validationData)

# calculam acuratetea si matricea de confuzie
accuracy = accuracy_score(validationLabels, predictedLabels)
confusionMatrix = confusion_matrix(validationLabels, predictedLabels)
confusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
confusionMatrixDisplay.plot()

print("Accuracy on validation data:", accuracy)


# In[6]:


# luam datele de testare
df_test = pd.read_csv("sample_submission.txt", dtype={'id': 'string'})
df_test['id'] = 'data/' + df_test['id'] + '.png'
testPaths = list(df_test['id'])
testData = np.array(list(image_generator(testPaths, len(testPaths))))
testData = testData.reshape(len(testPaths), -1)
testData = (testData - np.mean(testData)) / np.std(testData)

# prezicem etichetele pt imaginile de testare
predLabels = model.predict(testData)

# scriem etichetele in fisier
with open("predictionsSVM.txt", 'w') as f:
    f.write("id,class\n")
    for i in range(len(predLabels)):
        f.write(list(df_test['id'])[i].replace('data/', '').replace('.png', '') + ',' + str(predLabels[i]) + '\n')


# In[ ]:




