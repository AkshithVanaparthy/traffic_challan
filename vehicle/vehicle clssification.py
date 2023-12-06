import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load COCO labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open video file
video_file = 'red light.mp4'
cap = cv2.VideoCapture(video_file)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Preprocess the frame for the model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass
    outs = net.forward(layer_names)

    # Initialize lists for detected objects, their bounding boxes, and confidence scores
    class_ids = []
    confidences = []
    boxes = []

    # Set a confidence threshold to filter out weak detections
    confidence_threshold = 0.5

    # Loop through the outputs and collect detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Scale the bounding box coordinates to match the original frame size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in indices:
        # i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with object detection
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
