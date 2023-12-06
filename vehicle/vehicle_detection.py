import cv2
import numpy as np
from helmetdetection import helmet_detection
from plate import plate_detection
import red_light_violation
import database
import triple_detection
net = cv2.dnn.readNet('C:\\Users\\AKSHITH\\Documents\\traffic challan\\vehicle\\yolov3.cfg',
                      'C:\\Users\\AKSHITH\\Documents\\traffic challan\\vehicle\\yolov3.weights')

# Load COCO labels
with open('C:\\Users\\AKSHITH\\Documents\\traffic challan\\vehicle\\coco.names', "r") as f:
    classes = f.read().strip().split("\n")
def detect(frame,helmetviolated,redviolated,tripleviolated):

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
        #helmet violation
        if label=="motorbike":
            bike = frame[int(y*0.75):y+h,int(x*0.75):x+w]
            helmet_violate = helmet_detection.detect(bike)
            triple_violate = triple_detection.detect(bike)
            if helmet_violate:
                plate_number = plate_detection.detect(bike)
                if plate_number not in helmetviolated and plate_number!=None:
                    helmetviolated.append(plate_number)
                    database.insert_data(plate_number, 0, 1, 0,0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if triple_violate>=3:
                plate_number = plate_detection.detect(bike)
                if plate_number not in tripleviolated and plate_number!=None:
                    tripleviolated.append(plate_number)
                    database.insert_data(plate_number, 0, 0, 0,1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        centroid_y,centroid_x = (y+h)/2,(x+w)/2
        if red_light_violation.point_in_quadrilateral(centroid_x,centroid_y):
            veh = frame[y:y+h,x:x+w]
            plate_number = plate_detection.detect(veh)
            if plate_number not in redviolated and plate_number!=None:
                redviolated.append(plate_number)
                database.insert_data(plate_number, 1, 0, 0,0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        confidence = confidences[i]
        color = (0, 255, 0)  # Green color for bounding box
        #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame,helmetviolated,redviolated,tripleviolated
