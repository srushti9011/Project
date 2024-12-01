import cv2
import numpy as np
import face_recognition
import os
from twilio.rest import Client
import keys

# Initialize variables
classNames = []
encodeListKnown = []


# Load pre-trained encodings and class names
def load_known_encodings():
    path = 'Training_images'
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        classNames.append(os.path.splitext(cl)[0])
        encode = face_recognition.face_encodings(curImg)[0]
        encodeListKnown.append(encode)


# Load known encodings
load_known_encodings()
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert the frame from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        name = "Unknown"  # Default to unknown
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = classNames[best_match_index]

        # Draw a box around the face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # If the face is unknown (not in the training images), display "Unknown"
        if not any(matches):
            name = "Unknown"
            cv2.putText(frame, name, (left + 6, bottom + 15), font, 0.5, (255, 255, 255), 1)

            client = Client(keys.account_sid,keys.auth_token)

            msg_data = client.messages.create(
                body="someone is entering in your house",
                from_=keys.twilio_number,
                to=keys.target
            )    

    # Display the resulting image
    cv2.imshow('Webcam', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
