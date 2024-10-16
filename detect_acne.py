import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Load the acne detection model
model = load_model("facemodel.h5")

# Function to predict acne
def detect_acne(frame):
    # Preprocess the image for the model
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Make prediction
    (acne, withoutAcne) = model.predict(face)[0]
    return acne, withoutAcne

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the initial time
last_prediction_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the current time
    current_time = time.time()

    # Check if 5 to 10 seconds have passed
    if current_time - last_prediction_time >= 5:  # Adjust the time here (5 or 10 seconds)
        # Make prediction
        acne, withoutAcne = detect_acne(frame)
        label = "Acne" if acne > withoutAcne else "No Acne"
        confidence = max(acne, withoutAcne) * 100

        # Print the prediction result to the console
        print(f"Prediction: {label}, Confidence: {confidence:.2f}%")

        # Update the last prediction time
        last_prediction_time = current_time

    # Display the frame
    cv2.imshow("Acne Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
