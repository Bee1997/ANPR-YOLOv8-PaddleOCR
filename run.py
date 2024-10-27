import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
import csv
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Open the video file
video_path = "demoVideo.mp4"
cap = cv2.VideoCapture(video_path)

# Define the CSV file name
csv_file = "Car-List.csv"

# Video writing 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
video_write = cv2.VideoWriter('demoVideo_out.mp4',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 

# Set to keep track of processed car IDs
processed_ids = set()

ocr_flag = [0]*10
print("Program Started")
with open(csv_file, mode='w', newline='') as file:

    # Create a writer object
    writer = csv.DictWriter(file, fieldnames=["Car ID", "License Plate Number"])
    # Write the header
    writer.writeheader()

    start = time.time()

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True, conf=0.8, classes=[2], iou=0.3, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        for result in results[0].boxes:

            if result.id is not None:
                # Extract the track ID
                track_id = int(result.id)
                
                # Check if this car ID has already been processed for OCR
                if track_id not in processed_ids or ocr_flag[track_id]==False:
                    # Extract the bounding box coordinates for each detected car
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    car_roi = frame[y1:y2, x1:x2]

                    # Perform OCR on the detected car region
                    ocr_result = ocr.ocr(car_roi, cls=True)

                    # Check if OCR detected any text
                    if ocr_result and isinstance(ocr_result[0], list):
                        for line in ocr_result[0]:
                            # Extract the bounding box coordinates and text
                            box = np.array(line[0], dtype=np.int32) + np.array([x1, y1])  # Adjust box to the full frame coordinates
                            text = line[1][0]
                            score = line[1][1]

                            # Filter by score
                            if 95.0 < (score * 100.0):

                                # Draw the bounding box on the frame
                                cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

                                # Calculate the position to put the text (bottom-left corner of the box)
                                x, y = box[0][0], box[0][1]

                                ocr_flag[track_id]=True

                                writer.writerow({"Car ID": track_id, "License Plate Number": text})

                                # # Draw the text on the frame
                                text_position = (x, y - 10)
                                cv2.putText(annotated_frame, f'{text} ({score:.2f})', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # Add this car ID to the set of processed IDs
                    processed_ids.add(track_id)

        # Write the frame into the outputfile
        video_write.write(annotated_frame) 

        # Display the annotated frame
        cv2.imshow("ANPR using YOLOv8 + PaddleOCR", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

print("Total time: ", ((time.time())-start))
print("Program Stopped")

# Release the video capture and writing and close the display window
cap.release()
video_write.release() 
cv2.destroyAllWindows()
