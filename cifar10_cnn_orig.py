import tensorflow as tf
#import keras
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Activation
#from keras.layers import Dense, Dropout, Flatten
#from keras import datasets
from tensorflow.keras import models, layers, datasets
import numpy as np
import pandas as pd
import requests as re
from sklearn import metrics
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(train_images.shape)   #(50000, 32, 32, 3)
print(train_labels.shape)   #(50000, 1)
#normalize pixel values to be between 0 and 1
train_images, test_images =  train_images/255.0, test_images/255.0

class_names = ['airplane', "automobile","bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = None)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#configure CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.summary()

# add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()


#compile and train the model

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam', metrics= ['accuracy'])

history=model.fit(train_images, train_labels, epochs=10, verbose=1,
                validation_data=(test_images, test_labels))

#Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accury')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.legend(loc='upper left')
plt.show()

print("fit done")
#print test_loss, test_acc
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("%s: %f" %('test_loss', test_loss))
print("%s: %f" %('test_acc', test_acc))

# save configure
model_json = model.to_json()
with open("cnnimag0.config","w") as json_file:
    json_file.write(model_json)
print("save cnnimg0.config")

model.save_weights("cnnimag0.h5")
print("save cnnimag0.h5")

#later....
json_file = open("cnnimag0.config", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cnnimag0.h5")
print("loaded model configure & weights")

#re compile & evaluate
loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam', metrics= ['accuracy'])

test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)

print("%s: %f" %('test_loss', test_loss))
print("%s: %f" %('test_acc', test_acc))