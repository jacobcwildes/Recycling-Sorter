#This is the main body of the recycling sorter program

#Import requirements
from actuateServo import actuateServo
from tft import operateTFT

#Debug variable
material = "paper"
matType = "paper"

actuateServo(material)
operateTFT(matType)
