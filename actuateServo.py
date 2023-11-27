#Purpose: A simple function to actuate a servo motor to indicate 
#what sort of recyling is being viewed
#Import required libraries
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD) #Interface via physical pin number

pin = 12

GPIO.setup(pin, GPIO.OUT) #Select pin 12 as an output since it has hardware support for PWM

#Straight ahead is "Other." 
#45* to the right is "Plastic"
#45* to the left is "Paper
#90* to the right is "Glass"
#90* to the left is "Metal"

def actuateServo(material):
    if material == "plastic":
        servoPos = GPIO.PWM(pin, 500) #Period of 2mS tells servo to move to 45* angle
        servoPos.start(50) #Arbitrarily set PWM duty cycle to 50%
        sleep(5) #Wait to make sure that the servo has moved to position
        servoPos.stop() #Close the PWM connection
        GPIO.cleanup() #Close GPIO connection

    elif material == "paper":
        servoPos = GPIO.PWM(pin, 1000) #Period of 1.5mS tells servo to move to -45* angle
        servoPos.start(50) #Set PWM DC to 50%
        sleep(5) #Wait 5 seconds to make sure servo has moved to position
        servoPos.stop() #Close PWM connection
        GPIO.cleanup() #Close GPIO connection

    elif material == "glass":
        servoPos = GPIO.PWM(pin, 400) #Period of 2.5mS tells servo to go to 90* angle
        servoPos.start(50) #Set PWM DC to 50%
        sleep(5) #Wait 5 seconds to make sure the servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #CLose GPIO connection

    elif material == "metal":
        servoPos = GPIO.PWM(pin, 2000) #Period tells servo to go to -90* angle
        servoPos.start(50) #Set PWM to 50%
        sleep(5) #Make sure servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #Close GPIO connection

    else:
        servoPos = GPIO.PWM(pin, 666.6) #Period of 1.5mS tells servo to go to 0* angle
        servoPos.start(50) #Set PWM to 50%
        sleep(5) #Make sure servo moved
        servoPos.stop() #Close PWM
        GPIO.cleanup() #Close GPIO connection
       
GPIO.cleanup() #Make sure the GPIO is released
