#Purpose: A simple function to actuate a servo motor to indicate 
#what sort of recyling is being viewed
#Import required libraries
import RPi.GPIO as GPIO
import time

def actuateServo(material):

   # GPIO.setmode(GPIO.BOARD) #Interface via physical pin number

    pin = 12

    GPIO.setup(pin, GPIO.OUT) #Select pin 12 as an output since it has hardware support for PWM

    #Straight ahead is "Other." 
    #45* to the left is "Plastic"
    #45* to the right is "Paper
    #90* to the left is "Glass"
    #90* to the right is "Metal"

    if material == "plastic":
        servoPos = GPIO.PWM(pin, 50) #Need total pulse length of 20mS
        servoPos.start(10) #Duty cycle of 2 mS goes to -45*
        time.sleep(1) #Wait to make sure that the servo has moved to position
        servoPos.stop() #Close the PWM connection
        GPIO.cleanup() #Close GPIO connection

    elif material == "paper":
      #  print("Paper")
        servoPos = GPIO.PWM(pin, 50) #Pulse width of 20mS
        servoPos.start(5) #Duty cycle of 1 mS moves to 45* angle
        time.sleep(1) #Wait 5 seconds to make sure servo has moved to position
        servoPos.stop() #Close PWM connection
        GPIO.cleanup() #Close GPIO connection

    elif material == "glass":
        servoPos = GPIO.PWM(pin, 50) #Pulse width of 20mS
        servoPos.start(12.5) #Duty cycle of 2.5mS moves to -90* angle
        time.sleep(1) #Wait 5 seconds to make sure the servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #CLose GPIO connection

    elif material == "metal":
        servoPos = GPIO.PWM(pin, 50) 
        servoPos.start(2.5) #Go to 90* angle
        time.sleep(1) #Make sure servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #Close GPIO connection

    else:
        servoPos = GPIO.PWM(pin, 50)
        servoPos.start(7.5) #Go to 0* angle
        time.sleep(1) #Make sure servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #Close GPIO connection
       
#GPIO.cleanup() #Make sure the GPIO is released
