#Import needed modules
from picamera2 import Picamera2
import cv2 as cv

#Make a camera structure
cap = Picamera2()

#Pass some basic configurations
cap.configure(cap.create_still_configuration(main={"format": "XRGB8888", "size": (640, 480)}))

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
		cv.waitKey(1)
