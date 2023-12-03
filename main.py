#This is the main body of the recycling sorter program

#Import requirements
from actuateServo import actuateServo
from tft import operateTFT
import RPi.GPIO as GPIO

# Import for proximity sensor 
from proximitySensor import prox_sensor
import time 

# Import for the camera
from picamera2 import Picamera2
import cv2 as cv
import os

#import for model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2 as cv
import PIL.Image
import numpy as np
from audio_processing import get_spectrogram

# GPIO pins



# Camera

#Make a camera structure
cap = Picamera2()

#Pass some basic configurations
cap.configure(cap.create_still_configuration(main={"format": "RGB888", "size": (640, 480)}))
material = ""
while True:

	distance = prox_sensor()
	if distance is not None:
	# We can change the proximity sensor reading distance here
		if distance < 10.0:
			print("Recording will start")
			#Recording audio			
			cmd= "arecord -D hw:2,0 -d 4 -f cd ~/Desktop/Recycling-Sorter/test.wav -c 1"
			os.system(cmd)
			print("Recording finished")
			
			# Audio classification
			audioNet = tf.keras.models.load_model("model_audio.h5")
			x = "test.wav"
			x = tf.io.read_file(str(x))
			x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=70000,)
			x = tf.squeeze(x, axis=-1)
			waveform = x
			x = get_spectrogram(x)
			x = x[tf.newaxis,...]

			prediction = audioNet.predict(x)
			x_labels = ['glass', 'metal', 'paper', 'plastic']
			all_classes_audio = tf.nn.softmax(prediction[0])
			predicted_class_audio = np.argmax(all_classes_audio)
			label = f"Predicted class: {predicted_class_audio}"
			if predicted_class_audio == 0:
				material = "glass"
			elif predicted_class_audio == 1:
				material = "metal"
			elif predicted_class_audio == 2:
				material = "paper"
			else:
				material == "plastic"
			print(material)	
			
			# Camera logic here ------------------------
			#Begin the stream
			cap.start()
			
			#start function returns 0 on success, 1 on error
			#Make sure that the stream actually opened
			if cap.start():
				print("Unable to open camera stream")

			else:	
				
				#im = cap.capture_array()
				#print(im.shape)
				
				cap.capture_file("capture.jpg")
				
				# Call the model for the image segmentation
				ImageNet = tf.keras.models.load_model("image_classifier_model_4.h5")
				im = cv.imread('capture.jpg')
				#im = im.PIL.Image.convert('RGB')
				resized_frame = cv.resize(im, (224, 224))
			
				#Normalize pixel value:
				normalized_frame = resized_frame / 255.0
				
				input_image = np.expand_dims(normalized_frame, axis=0)
				
				prediction = ImageNet.predict(input_image)
				all_classes_img = tf.nn.softmax(prediction[0])
				predicted_class_img = np.argmax(all_classes_img)
				#predicted_class = np.argmax(prediction, axis=1)[0]
				
				label = f"Predicted class: {predicted_class_img}"
				
				#cv.putText(im, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				
				
				#if cv.waitKey(1) & 0xFF == ord('q'):
					# Take a picture for segmentation purposes
				#	cv.imwrite('images/c1.png', im)
				cv.destroyAllWindows()
				#	break
			# Camera logic finishes here ------------------------
			# Change the prediction from number to name for image classification
			if predicted_class_img == 0:
				material = "metal"
			elif predicted_class_img == 1:
				material = "plastic"
			elif predicted_class_img == 2:
				material = "paper"
			else:
				material == "glass"
			print(material)	
			
			# Ensemble models
			all_classes = np.zeros(4)
			all_classes[0] = (all_classes_img[0] + all_classes_audio[1] ) / 2.0
			all_classes[1] = (all_classes_img[1] + all_classes_audio[3] ) / 2.0
			all_classes[2] = (all_classes_img[2] + all_classes_audio[2] ) / 2.0
			all_classes[3] = (all_classes_img[3] + all_classes_audio[0] ) / 2.0
			print("All possibilities", all_classes)	
			predicted_class = np.argmax(all_classes)
		
			if predicted_class == 0:
				material = "metal"
			elif predicted_class == 1:
				material = "plastic"
			elif predicted_class== 2:
				material = "paper"
			else:
				material == "glass"
			print(material)	
			print("overall prediction: ", predicted_class)
			# Print out the garbage material type
			
			# Show the corresponding label in TFT
			operateTFT(material)
			
			# Turn the servo accordingly
			actuateServo(material)
			
		
		
		
		





