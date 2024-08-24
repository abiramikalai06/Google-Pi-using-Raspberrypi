import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Get list of available voices
voices = engine.getProperty('voices')

# Select the desired voice (change index as needed)
engine.setProperty('voice', voices[1].id)  # Change index as needed

# Initialize OpenCV VideoCapture object for webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam, you can change it if you have multiple webcams
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

print("Camera opened successfully.")

# Load your trained model
try:
    model = load_model('Path\model.keras')
except Exception as e:
    print("Error loading the model:", str(e))
    exit()
    
# Dictionary mapping class indices to currency denominations
class_labels = {0: '10', 1: '20', 2: '50', 3: '100', 4: '200', 5: '500'}

def preprocess_frame(frame, target_size=(192, 192)):
    # Resize the frame to the target size
    resized_frame = cv2.resize(frame, target_size)

    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame.astype('float32') / 255.0
    
    # Expand dimensions to match model input shape
    input_frame = np.expand_dims(normalized_frame, axis=0)
    
    return input_frame

def detect_currency(frame):
    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Apply your currency detection model to the preprocessed frame
    predictions = model.predict(input_frame)
    
    # Convert predictions to currency label
    currency_label_index = np.argmax(predictions)  # Get the index of the class with the highest probability
    print("predicted class index:", currency_label_index)
    currency_label = class_labels[currency_label_index]  # Convert index to currency label using class labels
    print("predicted label:", currency_label)

    return currency_label

# Main loop for currency detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform currency detection
    currency_label = detect_currency(frame)

    if currency_label is not None:
        cv2.putText(frame, f'INR {currency_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Currency Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources
cap.release()
cv2.destroyAllWindows()
