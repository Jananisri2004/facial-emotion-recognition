from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from proj import CNN  # Import your CNN model from proj.py

app = Flask(__name__)

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

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    cap = cv2.VideoCapture(0)

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
            # Extract the face region and perform emotion recognition
            face_gray = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face_gray, (48, 48))
            pil_img = transform(resized_face).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                outputs = model(pil_img)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = emotion_labels[predicted.item()]

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the predicted emotion label on the frame
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
