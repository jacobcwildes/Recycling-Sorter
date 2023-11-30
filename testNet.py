import tensorflow as tf
import cv2 as cv
import PIL.Image
import numpy as np

cap = cv.VideoCapture(0)

ImageNet = tf.keras.models.load_model("Models/ImageNet0.9116.keras")
#ImageNet.summary()



if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame. Exiting")
        break
    resized_frame = cv.resize(frame, (150, 150))
    
    #Normalize pixel value:
    normalized_frame = resized_frame / 255.0
    
    input_image = np.expand_dims(normalized_frame, axis=0)
    
    prediction = ImageNet.predict(input_image)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    label = f"Predicted class: {predicted_class}"
    cv.putText(frame, label, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv.imshow('Video Stream', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
       
