import RPi.GPIO as GPIO
import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# width = int(cap.get(3))
# height = int(cap.get(4))



while True:
	ret, frame = cap.read()
	cv2.imshow('Deneme', frame)
	if cv2.waitKey(1) == ord('q'):
		break
	
	cap.release()
	cv2.destroyAllWindows()
        
    else:
        print ("The door is secure")
