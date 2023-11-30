#This is the main body of the recycling sorter program

#Import requirements
from actuateServo import actuateServo
from tft import operateTFT
import RPi.GPIO as GPIO

# Import for proximity sensor 
from proximitySensor import prox_sensor

# Import for the camera
from picamera2 import Picamera2
import cv2 as cv
import os

# GPIO pins



# Camera

#Make a camera structure
cap = Picamera2()

#Pass some basic configurations
cap.configure(cap.create_still_configuration(main={"format": "XRGB8888", "size": (640, 480)}))

actuateServo("paper")

while True:

	distance = prox_sensor()
	# We can change the proximity sensor reading distance here
	if distance < 10:
		
		print("Recording will start")
		#Recording audio			
		cmd= "arecord -D hw:2,0 -d 5 -f cd ~/Desktop/471/project/test.wav -c 1"
		os.system(cmd)
		
		# Camera logic here ------------------------
		#Begin the stream
		cap.start()

		#start function returns 0 on success, 1 on error
		#Make sure that the stream actually opened
		if cap.start():
			print("Unable to open camera stream")

		else:	
			while True:
				im = cap.capture_array()
				cv.imshow("Frame", im)
				if cv.waitKey(1) & 0xFF == ord('q'):
					# Take a picture for segmentation purposes
					cv.imwrite('images/c1.png', im)
					cv.destroyAllWindows()
					break
		# Camera logic finishes here ------------------------
		
		
		# Call the model for the image segmentation
		
		# Call the model for the audio recognition
		
		# Ensemble models
		
		# Print out the garbage material type
		
		# Show the corresponding label in TFT
		#operateTFT(matType)
		
		# Turn the servo accordingly
		#actuateServo(material)
		
		
		
		
		





