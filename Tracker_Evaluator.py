#-------------------------------------------------------------------------------
# IMPORT PACKAGES

import cv2
import sys
import time
from xml.dom import minidom

#-------------------------------------------------------------------------------
# DEFINE VARIABLES

count = 1
total = 0
pen = 0
TruePos = 0
TrueNeg = 0
FalseNeg = 0
FalsePos = 0
total_time = 0

#-------------------------------------------------------------------------------
# SETUP TRACKER

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__' :

    # Creator tracker
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

    # Read first image
    image = cv2.imread("eval_images/image" + str(count)+".png")
	# Open xml doc and get GT coordinates
    xmldoc = minidom.parse("eval_images/image" + str(count)+".xml")
    GT_xmin = int(xmldoc.getElementsByTagName("xmin")[0].firstChild.data)
    GT_ymin = int(xmldoc.getElementsByTagName("ymin")[0].firstChild.data)
    GT_xmax = int(xmldoc.getElementsByTagName("xmax")[0].firstChild.data)
    GT_ymax = int(xmldoc.getElementsByTagName("ymax")[0].firstChild.data)

    # Define an initial bounding box with GT bounding box
    x_width = abs(GT_xmax-GT_xmin)
    y_height = abs(GT_ymax-GT_ymin)
    bbox = (GT_xmin, GT_ymin, x_width, y_height)
    frame = cv2.rectangle(image, (GT_xmin,GT_ymax), (GT_xmax, GT_ymin), (255, 255, 255), 3)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    count += 1

	# Until all frames have been read
    while count < 388:
        no_pen = False
        # Read a new frame
        frame = cv2.imread("eval_images/image" + str(count)+".png")
        # Start timer
        timer = cv2.getTickCount()
        t0 = time.time()
        # Update tracker
        ok, bbox = tracker.update(frame)
        t1 = time.time()
        execute_time = t1-t0
        total_time = total_time + execute_time
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Draw bounding box for GT
        xmldoc = minidom.parse("eval_images/image" + str(count)+".xml")
        GT_xmin = int(xmldoc.getElementsByTagName("xmin")[0].firstChild.data)
        GT_ymin = int(xmldoc.getElementsByTagName("ymin")[0].firstChild.data)
        GT_xmax = int(xmldoc.getElementsByTagName("xmax")[0].firstChild.data)
        GT_ymax = int(xmldoc.getElementsByTagName("ymax")[0].firstChild.data)
        cv2.rectangle(frame, (GT_xmin, GT_ymin), (GT_xmax, GT_ymax), (255, 255, 255), 2)
        # Draw bounding box for ST
		(x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ST_xmin = x
        ST_ymin = y
        ST_xmax = x + w
        ST_ymax = y + h
		# Calculate centre distance and components for overlap
        dx = abs(min(ST_xmax, GT_xmax)-max(ST_xmin, GT_xmin))
        dy = abs(min(ST_ymax, GT_ymax)-max(ST_ymin, GT_ymin))
        GT_centre = GT_xmin+(1/2)*(GT_xmax-GT_xmin)
        ST_centre = ST_xmin+(1/2)*(ST_xmax-ST_xmin)
        centre_dist = abs(ST_centre-GT_centre)

        overlap_area = dx*dy
        ST_area = (ST_xmax-ST_xmin)*(ST_ymax-ST_ymin)
        GT_area = (GT_xmax-GT_xmin)*(GT_ymax-GT_ymin)

        if (ST_xmin == 0)*(ST_ymin == 0)*(ST_xmax == 0)*(ST_ymax == 0):
		# no detection made
            no_pen = True
            if GT_area <= 20:
                #True Negative
                cond = "True Negative"
                pen = pen + 0
                TrueNeg += 1
            else:
                #False Negative
                cond = "False Negative"
                pen = pen + 1
                FalseNeg += 1
        elif ((ST_xmin != 0)+(ST_ymin != 0)+(ST_xmax != 0)+(ST_ymax != 0))*(GT_area <= 20):
        # Detection has been made and target is not occluded
			overlap_area = 0
        else:
		# Calculate IOU to use later   
            union = abs(GT_area+ST_area-overlap_area)
            IOU = overlap_area/union

        if no_pen == False:
		# if no penalties have been applied
            if (IOU > 0.7)*(centre_dist < 20) + (centre_dist < 10):
                # True Positive
                TruePos += 1
                cond = "True Positive"
                pen = pen + (1-IOU)
            elif (IOU > 4):
                #False Positive
                FalsePos += 1
                cond = "False Positive"
                pen = pen + 3*(1-IOU)
            else:
                #False Positive
                FalsePos += 1
                cond = "False Positive"
                pen = pen + 3*(1-IOU)


        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if q pressed
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (count > 100)*(count<290):
            cv2.waitKey(0)

#print("Total Penalty: ", pen)
#print("True Positive:  ", TruePos)
#print("False Positive: ", FalsePos)
#print("True Negative:  ", TrueNeg)
#print("False Negative:  ", FalseNeg)
#print("Average Time: ", total_time/(count-2))
cv2.destroyAllWindows()
