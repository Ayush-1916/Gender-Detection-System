import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN

# The full, updated code for real-time gender detection using EfficientNetB3 and MTCNN.
# This code is perfectly aligned with your training script.
#/Users/ayushkoge/gender_13k/gender_detection_efficientnetb3(20ep).keras
# Load the trained EfficientNetB3 model
# Ensure the path is correct
model = load_model("/Users/ayushkoge/gender_13k/gender_detection_efficientnetb3(20ep).keras")

# Initialize the MTCNN face detector
detector = MTCNN()

# Open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man', 'woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # Convert the webcam frame to RGB format for MTCNN
    # This is a critical step, as MTCNN expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MTCNN to detect faces
    faces = detector.detect_faces(rgb_frame)

    # loop through detected faces
    for result in faces:

        # Get corner points of face rectangle from MTCNN output
        x, y, w, h = result['box']
        confidence = result['confidence']

        # Ensure confidence is high enough before processing
        if confidence > 0.95:

            # Get the coordinates for the rectangle
            startX, startY = x, y
            endX, endY = x + w, y + h

            # draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face region
            face_crop = np.copy(rgb_frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # Preprocessing for EfficientNetB3 model
            # 1. Resize the image to 300x300
            face_crop = cv2.resize(face_crop, (300, 300))
            
            # 2. Normalize pixel values
            face_crop = face_crop.astype("float") / 255.0
            
            # 3. Convert to array and expand dimensions
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            # The model was trained on a single channel output for binary classification
            conf = model.predict(face_crop, verbose=0)[0] 
            
            # get label with max accuracy
            idx = np.argmax(conf) # This will always be 0 as there's only 1 output
            
            # We must check the output value of the sigmoid layer
            if conf[0] > 0.7:
                # The model outputs a value between 0 and 1.
                # If it's > 0.5, it's 'woman', otherwise it's 'man'.
                label = 'man'
                conf_score = conf[0]
            else:
                label = 'woaman'
                conf_score = 1 - conf[0]

            label_text = "{}: {:.2f}%".format(label, conf_score * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    # display output
    cv2.imshow("Gender Detection (EfficientNetB3)", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()