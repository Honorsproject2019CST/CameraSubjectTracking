#--------------------------------------------------------------------------------
# IMPORT PACKAGES

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import serial

#--------------------------------------------------------------------------------
# DEFINE VARIABLES

end = False
count = 0
total_time = 0
read_time = 0
update_time = 0
serial_time = 0

#--------------------------------------------------------------------------------
# SERIAL SETUP

ser = serial.Serial('COM7', 115200) # Establish the connection on a specific port

#--------------------------------------------------------------------------------
# TRACKER SETUP

# Check version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
 
    # Set up tracker.
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

#--------------------------------------------------------------------------------
# DETECTOR SETUP

# Add the path to the current folder
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the frozen inference graph
MODEL_NAME = 'inference_graph_ssdlite'

# Path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects..
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#--------------------------------------------------------------------------------
# TRACKER

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)
n = 0
total_time = 0

while(True):
    start = time.time()     # Get current time
    ret, frame = video.read()   # Get newest frame
    frame_expanded = np.expand_dims(frame, axis=0)  # expand frame

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.70)
    # Get coordinates
    ymin = int((boxes[0][0][0]*frame.shape[0]))
    xmin = int((boxes[0][0][1]*frame.shape[1]))
    ymax = int((boxes[0][0][2]*frame.shape[0]))
    xmax = int((boxes[0][0][3]*frame.shape[1]))
    conf_score = float(scores[0][0])*100
    print("Score:", str(conf_score))

    # Display frame
    cv2.imshow('Object detector', frame)
    end = time.time()   # Get current time
    time_taken = end-start
    total_time = total_time + time_taken
    n = n+1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        end = True
        break
    # If confidence is greater than threshold, begin tracking
    if conf_score >= 95:
        print("Object Detected!")
        end = False
        break

if end == False:    # begin tracking
    bbox = (xmin, ymin, xmax-xmin, ymax-ymin)   # use object location from detector
    ok = tracker.init(frame, bbox)  # initialise tracker
    if ok == True:
        print("Init Success")   # notify if initialisation was successful
    else:
        end == True # end program if failed
    while end == False:
        t0 = time.time()
        ok, frame = video.read()    # read frame
        t1 = time.time()
        ok, bbox = tracker.update(frame)    # update tracker
        t2 = time.time()
        if ok:
            # Tracking success
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            t3 = time.time()
            x_trans = str(x + int(w/2)) #send centre point to arduino
            # add preceding zeros up to 2
            if len(x_trans) == 1:
                x_trans = "0" + "0" + x_trans
            if len(x_trans) == 2:
                x_trans = "0" + x_trans
            if len(x_trans) == 3:
                x_trans = x_trans
            ser.write(str.encode(x_trans))  #write serial data to port
            t4 = time.time()
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        # Display frame
        cv2.imshow("Object detector", frame)
        # If q is pressed, exit
        if cv2.waitKey(1) & 0xff ==ord('q'):
            end = True
        t5 = time.time()
##        read_time = read_time + (t1-t0)
##        update_time = update_time + (t2-t1)
##        serial_time = serial_time + (t4-t3)
##        total_time = total_time + (t5-t0)
        count += 1
    
# Clean up
if end == True:
    video.release()
    cv2.destroyAllWindows()
    avg_read = read_time/count
    avg_update = update_time/count
    avg_serial = serial_time/count
    avg_time = total_time/count
##    print("Average Time: ",avg_time)
##    print("Average Read Time: ",avg_read)
##    print("Average Update Time: ",avg_update)
##    print("Average Serial Time: ",avg_serial)

    

