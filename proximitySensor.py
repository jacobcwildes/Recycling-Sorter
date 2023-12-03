# Import for proximity sensor 
import RPi.GPIO as GPIO
import time

def prox_sensor():
	try:
		
		# Proximity sensor
		GPIO.setmode(GPIO.BCM)
		#GPIO.setwarnings(False)
		TRIG = 18
		ECHO = 6
		GPIO.setup(TRIG, GPIO.OUT)
		GPIO.setup(ECHO, GPIO.IN)
		counter = 0 
		# Proximity sensor logic
		#time.sleep(0.5)
		GPIO.output(TRIG, True)
		time.sleep(0.00001)
		GPIO.output(TRIG, False)
		while GPIO.input(ECHO) == 0:
			pulse_start = time.time()
		
		while GPIO.input(ECHO) == 1:
			pulse_end = time.time()
			
		GPIO.cleanup()
		pulse_duration = pulse_end - pulse_start
		distance = pulse_duration * 17150
		distance = round(distance,2) 
		print("Distance is : ", distance)
		GPIO.cleanup()
		return distance	
	except Exception as e:
		print("An error occured:")
		
		time.sleep(1)
		return None
