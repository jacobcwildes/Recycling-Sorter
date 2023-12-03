import cv2
from pathlib import Path
import numpy as np
index = -1
cap = cv2.VideoCapture(-1)
cap.set(3, 640)
cap.set(4, 480)
#width = int(cap.get(3))
#height = int(cap.get(4))


while True:
	ret, frame = cap.read()
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
