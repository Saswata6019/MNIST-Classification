import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

import PIL

from PIL import Image

import numpy as np

from numpy import asarray

# Data load and visualization of a part of the dataset

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.get_cmap('gray'))
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# Data preparation # The reshape has been done to nullify a dimension identification bug

train_images = train_images.reshape(-1, 28, 28, 1)

test_images = test_images.reshape(-1, 28, 28, 1)

print("The training images have a shape of {} and the test images have a shape of {}".format(
	train_images.shape, test_images.shape))

# Normalize pixel values to be between 0 and 1

train_images, test_images = train_images / 255.0, test_images / 255.0

# Model preparation and summary

classifier = models.Sequential()

classifier.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

classifier.add(layers.MaxPooling2D((2, 2)))

classifier.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(layers.MaxPooling2D((2, 2)))

classifier.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(layers.Flatten())

classifier.add(layers.Dense(100, activation = 'relu'))

classifier.add(layers.Dense(10, activation = 'softmax'))

classifier.summary()

# Training and evaluation

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(train_images, train_labels, batch_size = 32, epochs = 10, verbose = 1, 
	validation_data = (test_images, test_labels))

# Plotting

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(history.history['acc'], label = 'Training Accuracy') # In case an error arises, try changing 'acc' to the value 
# specifed in the metrics argument of the classifier.compile function, which in this case is, 'accuracy'.

plt.plot(history.history['val_acc'], label = 'Validation Accuracy') # In case an error arises, try changing 
# from 'val_acc' to 'val_accuracy'. Such problems, like the previous one might arise due to version differences.

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc = 'lower right')

plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label = 'Training Loss') 

plt.plot(history.history['val_loss'], label = 'Validation Loss') 

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc = 'upper right')

plt.title('Training and Validation Loss')

plt.show()

test_loss, test_acc = classifier.evaluate(test_images, test_labels, verbose = 1)

print("The accuracy of the model is {} and the loss associated with the model is {}".format(test_acc, test_loss))

# Prediction 

def load_image(filename):

	img = Image.open(filename)

	img = img.convert(mode = 'L')

	img = img.resize((28, 28))

	img = asarray(img)

	img = img.reshape(-1, 28, 28, 1)

	img = img / 255.0

	return img

img = load_image('D:/Python Programs/Neural Network/sample_image.jpg')

digit = classifier.predict_classes(img)

print("The given image is of the number {}".format(digit[0]))




