from plate import plate_detection
import cv2
import numpy as np
import time
from speeding import speed
from vehicle import vehicle_detection
video_file = 'plate.mp4'
cap = cv2.VideoCapture(video_file)
helmetviolated = []
redviolated = []
speedviolated = []
tripleviolated = []
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(plate_detection.detect(frame))

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
