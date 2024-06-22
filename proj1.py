import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from proj import CNN  # Import your CNN model from proj.py

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('emotion_model.pth'))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Define emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face

        # Extract the face region and perform emotion recognition
        face_gray = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_gray, (48, 48))
        pil_img = transform(resized_face).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(pil_img)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = emotion_labels[predicted.item()]

        # Display the predicted emotion label on the frame
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()





