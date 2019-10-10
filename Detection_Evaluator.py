#-------------------------------------------------------------------------------
# IMPORT PACKAGES
import os
import cv2
import time
import numpy as np
import tensorflow as tf
import sys
import csv
from xml.dom import minidom

#-------------------------------------------------------------------------------
# DFINE VARIABLES

count = 1
pen = 0
TruePos = 0
TrueNeg = 0
FalseNeg = 0
FalsePos = 0
total_time = 0

#-------------------------------------------------------------------------------
# SETUP DETECTOR

# Add the path to the current folder
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the frozen inference graph
MODEL_NAME = 'inference_graph_ssdlite'
# Name of the directory containing the evaluation images
IMAGE_NAME = 'eval_images/image1.png'

# Path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

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

# Each score represents level of confidence for each of the objects.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Process all images
while count<388:
    pause = False
    detection_fail = False
    True_neg = False
    image = cv2.imread("eval_images/image" + str(count)+".png")
    image_expanded = np.expand_dims(image, axis=0)

    t0 = time.time()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    t1 = time.time()
    execute_time = t1-t0
    total_time = total_time + execute_time

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.60)

    conf = scores[0][0]*100	# Get confidence score
    if conf < 60:
        detection_fail = True
    else:
		# Get coordinates of bounding box for system track
        ST_ymin = int((boxes[0][0][0]*image.shape[0]))
        ST_xmin = int((boxes[0][0][1]*image.shape[1]))
        ST_ymax = int((boxes[0][0][2]*image.shape[0]))
        ST_xmax = int((boxes[0][0][3]*image.shape[1]))
	# Open xml doc for file
    xmldoc = minidom.parse("eval_images/image" + str(count)+".xml")
	# Get bounding box for ground truth
    GT_xmin = int(xmldoc.getElementsByTagName("xmin")[0].firstChild.data)
    GT_ymin = int(xmldoc.getElementsByTagName("ymin")[0].firstChild.data)
    GT_xmax = int(xmldoc.getElementsByTagName("xmax")[0].firstChild.data)
    GT_ymax = int(xmldoc.getElementsByTagName("ymax")[0].firstChild.data)
	# display image
    outimg = cv2.rectangle(image, (GT_xmin,GT_ymax), (GT_xmax,GT_ymin), (255,255,255), 2)
	# calculate compponents used in overlap area components
    dx = abs(min(ST_xmax, GT_xmax)-max(ST_xmin, GT_xmin))
    dy = abs(min(ST_ymax, GT_ymax)-max(ST_ymin, GT_ymin))
	# calculate distance between GT and ST centres 
    GT_centre = GT_xmin+(1/2)*(GT_xmax-GT_xmin)
    ST_centre = ST_xmin+(1/2)*(ST_xmax-ST_xmin)
    centre_dist = abs(ST_centre-GT_centre)

    overlap_area = dx*dy
    GT_area = (GT_xmax-GT_xmin)*(GT_ymax-GT_ymin)
    ST_area = (ST_xmax-ST_xmin)*(ST_ymax-ST_ymin)
    union = abs(GT_area+ST_area-overlap_area)
    IOU = overlap_area/union
    if detection_fail == True:
        if GT_area <= 20:
            #True negative
            cond = "True Negative"
            pause = False
            TrueNeg += 1
            pen = pen + 0
        else:
            #False negative
            cond = "False Negative"
            overlap_area = 0
            pause = False
            FalseNeg += 1
            pen = pen + 1
    else:
        if (IOU > 0.7)*(centre_dist < 20) + (centre_dist < 10):
            #True Positive
            cond = "True Positive"
            TruePos += 1
            pen = pen + (1-IOU)
        else:
            #False Positive
            cond = "False Positive"
            pause = False
            FalsePos += 1
            pen = pen + 3*(1-IOU)
    

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)
    len_count = len(str(count))
    print("Frame: ", count, " "*(3-len_count), "Category: ", cond)
        cv2.waitKey(0)
    count += 1
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    

# Press any key to close the image
#cv2.waitKey(0)

# Clean up
#print("Total Penalty: ", pen)
#print("True Positive:  ", TruePos)
#print("False Positive: ", FalsePos)
#print("True Negative:  ", TrueNeg)
#print("False Negative:  ", FalseNeg)
#print("Average Time: ", total_time/(count-1))
cv2.destroyAllWindows()
