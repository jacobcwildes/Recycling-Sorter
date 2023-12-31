import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735 
import RPi.GPIO as GPIO

def operateTFT(matType):
    picture = None
    if matType == "paper":
        picture = "images/misc_im/Paper.png"
    elif matType == "plastic":
        picture = "images/misc_im/ECE471_Final_TFT_Display_Plastic.jpg"
    elif matType == "metal":
        picture = "images/misc_im/ECE_471_Final_TFT_Display_Metal.jpg"
    else:
        picture = "images/misc_im/other-waste.jpg"
	
    GPIO.setmode(GPIO.BCM)
    cs_pin = digitalio.DigitalInOut(board.CE0)
    dc_pin = digitalio.DigitalInOut(board.D25)
    reset_pin = digitalio.DigitalInOut(board.D24)

    # Config for display baudrate (default max is 24mhz):
    BAUDRATE = 24000000
    
    
    

    # Setup SPI bus using hardware SPI:
    spi = board.SPI()

    if not spi:
        print("Error opening SPI device...")
        return -10

    # Create the display:
    disp = st7735.ST7735R(spi, rotation=90, 
					    cs=cs_pin,
					    dc=dc_pin,
					    rst=reset_pin, 
                        bgr=True, #TFT display correction (wired backward by default?)
					    baudrate=BAUDRATE) # 1.8" ST7735R

    
    if not disp:
        print("Error opening display...")
        return -10
    # Turn on the Backlight
    backlight = digitalio.DigitalInOut(board.D26)
    backlight.switch_to_output()
    backlight.value = True

    # Create blank image for drawing.
    # Make sure to create image with mode 'RGB' for full color.
    if disp.rotation % 180 == 90:
        height = disp.width  # we swap height/width to rotate it to landscape!
        width = disp.height
    else:
        width = disp.width  # we swap height/width to rotate it to landscape!
        height = disp.height
    image = Image.new("RGB", (width, height))


    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
    disp.image(image)

    image = Image.open(picture)
    #image.show()


    # Scale the image to the smaller screen dimension
    image_ratio = image.width / image.height
    screen_ratio = width / height
    if screen_ratio < image_ratio:
        scaled_width = image.width * height // image.height
        scaled_height = height
    else:
        scaled_width = width
        scaled_height = image.height * width // image.width
    image = image.resize((scaled_width, scaled_height), Image.BICUBIC)

    # Crop and center the image
    x = scaled_width // 2 - width // 2
    y = scaled_height // 2 - height // 2
    image = image.crop((x, y, x + width, y + height))

    # Display image.
    disp.image(image)


