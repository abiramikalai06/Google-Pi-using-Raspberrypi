import cv2
import numpy as np
import pyttsx3
import torch

# Check GPU availability
print("Num GPUs Available: ", torch.cuda.device_count())

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
else:
    print("Camera opened successfully.")

# Load pre-trained object detection model (e.g., YOLOv3)
net = cv2.dnn.readNet("D:\Final Project\OD\yolov3.weights", "D:\Final Project\OD\yolov3.cfg")  # Provide the paths to your YOLO model files
classes = []
with open("D:\Final Project\OD\coco.names", "r") as f:
    classes = f.read().splitlines()

# Set up text-to-speech engine
engine.setProperty('rate', 200)  # Adjust speech rate

while True:
    # Capture frame from camera
    ret, img = cap.read()
    if not ret:
        break

    # Perform object detection
    img = cv2.resize(img, (640, 480))
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate coordinates for bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Speak out the labels
    if isinstance(indexes, np.ndarray):
        for i in indexes.flatten():
            label = str(classes[class_ids[i]])
            engine.say(label)
            engine.runAndWait()

    # Draw bounding boxes and labels
    if isinstance(indexes, np.ndarray) and len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.waitKey(1)  # Add a short delay to give time for keyboard events
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Script completed successfully.")
