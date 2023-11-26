import os
import zipfile

import tensorflow.keras as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


datagen = ImageDataGenerator(
      rotation_range=100,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')  #Strategy to fill in new pixels as a result of transforming

#Augment the data so that the model won't see the same picture twice
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40, #Range in which to randomly rotate pictures
      width_shift_range=0.2, #Range to translate pictures horizontally/veritcally
      height_shift_range=0.2,
      shear_range=0.2,  #Randomly apply shear transformations
      zoom_range=0.2, ##Randomly zoom pictures
      horizontal_flip=True #Flip half images horizontally
      )

val_datagen = ImageDataGenerator(rescale=1./255) #Validation data does not get augmented
      
#Pull in training/verification data
base_dir = 'images'
train_dir = os.path.join(base_dir, 'ImageNet_Train')
validation_dir = os.path.join(base_dir, 'ImageNet_Test')


##TRAINING
#######################################
#Can train
train_can_dir = os.path.join(train_dir, 'cans')

#Paper train
train_paper_dir = os.path.join(train_dir, 'paper')

#Glass train
train_glass_dir = os.path.join(train_dir, 'glass')

#Plastic train
train_plastic_dir = os.path.join(train_dir, 'plastic')

train_can_fnames = os.listdir(train_can_dir)
#######################################

##TESTING
#######################################
#Can test
test_can_dir = os.path.join(validation_dir, 'cans')

#Paper test
test_paper_dir = os.path.join(validation_dir, 'paper')

#Glass test
test_glass_dir = os.path.join(validation_dir, 'glass')

#Plastic test
test_plastic_dir = os.path.join(validation_dir, 'paper')

#######################################

##Visualize datagen transformations on a subset of cars

img_path = os.path.join(train_can_dir, train_can_fnames[2])
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# The .flow() command below generates batches of randomly transformed images
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
  plt.figure(i)
  imgplot = plt.imshow(array_to_img(batch[0]))
  i += 1
  if i % 5 == 0:
    break
  
#Flow training images in batches of 20
# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=2,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
        
#Flow validation images in batches of 20
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=2,
        class_mode='categorical')
        
# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(7, activation='sigmoid')(x)

# Configure and compile the model
model = Model(img_input, output)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(0.00001),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=int(8/2),
      epochs=20,
      validation_data=validation_generator,
      validation_steps=int(4/2),
      verbose=2)
      

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))
print("Epochs: ", acc)
print("Acc: ", acc)
print("val_acc: ", val_acc)

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()
