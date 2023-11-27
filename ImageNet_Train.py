#Tensorflow dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import math

#Define a learning rate schedule function
def learning_rate_schedule(epoch):
	initial_lr = 0.001 #Initial learning rate
	drop = 0.5 #Learning rate drop factor
	epochs_drop = 5 #Number of epoxhs after which to drop learning rate
	lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
	return lr
	
#Define an early stop callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #Patience is how many epochs can elapse without improvement before the model stops training. Restore best weights will ensures we get the model with the best performance when training ends.

#Create learning rate scheduler callback
lr_scheduler = LearningRateScheduler(learning_rate_schedule)

#Set up train and validation directory
train_dir = 'images/ImageNet_Train'
validation_dir = 'images/ImageNet_Test'

#Define batch, image height and width 
img_width, img_height = 150, 150
batch_size = 1

#Create CNN model
model = Sequential([
	Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
	MaxPooling2D((2, 2)),
	
	Conv2D(64, (3, 3), activation='relu'),
	MaxPooling2D((2, 2)),
	
	#Conv2D(128, (3, 3), activation='relu'),
	#MaxPooling2D((2, 2)),
	
	#Conv2D(256, (3, 3), activation='relu'),
	#MaxPooling2D((2, 2)),
	
	Flatten(),
	Dense(128, activation='relu'),
	Dropout(0.5), #Dropout rate of 0.5
	
	Dense(4, activation='softmax') #Output layer with 4 units for 4 different categories
])

#Compile the model
optimizer = Adam(learning_rate=0.000001) #Initial learning rate
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

#Create image data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1./255, #Pixel values range from 1 to 255, scales pixels to between 0 and 1
	rotation_range = 120, #Rotate images by up to 120 degrees
	width_shift_range=0.5, #Shift images horizontally by up to 50% of their width
	height_shift_range=0.6, #Shift images vertically by up to 60% of their height
	shear_range=0.8, #Apply shear-based transforms
	zoom_range=0.9, #Randomly zoom images up to 20%
	horizontal_flip=True) #Flip images horizontally
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size = (img_width, img_height),
	batch_size = batch_size,
	class_mode = 'categorical')
	
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
	validation_dir,
	target_size = (img_width, img_height),
	batch_size = batch_size,
	class_mode = 'categorical')
	
#Train the model
history = model.fit(
	train_generator,
	steps_per_epoch=train_generator.samples // batch_size, #Floor division
	epochs=100,
	callbacks=[lr_scheduler, early_stopping],
	validation_data=validation_generator,
	validation_steps=validation_generator.samples // batch_size,
	verbose=1
	)
	
# Retrieve the training and validation accuracy values from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Print the training and validation accuracies for each epoch
for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), 1):
    print(f"Epoch {epoch}: Training Accuracy - {train_acc:.4f}, Validation Accuracy - {val_acc:.4f}")
    
#Export model:
model.save('Models/ImageNet' + str(f"{val_accuracy[-1]:.4f}") + '.keras')
