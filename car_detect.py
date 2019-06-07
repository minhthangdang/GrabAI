from config import config
import cv2
import numpy as np

def detect_bounding_box(image):
    (H, W) = image.shape[:2]

    # initialize yolo net
    print("[INFO] loading yolo network...")
    labels = open(config.YOLO_NAMES).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG, config.YOLO_WEIGHTS)

    # get last layer of the network for forward function later
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # normalize image for yolov3 net
    # 1. scale it to range of 0 to 1
    # 2. resize to (416, 416) without cropping
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # pass blob to network
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    # for each out layer
    for out in outs:
        # for each detection in out layer
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # if confidence greater than threshold and label is car
            if confidence > config.YOLO_MIN_CONFIDENCE and labels[class_id] == "car":
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width    = int(detection[2] * W)
                height   = int(detection[3] * H)
                x        = int(center_x - (width / 2))
                y        = int(center_y - (height / 2))
                class_ids.append(class_id)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    # apply non-maximum suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.YOLO_MIN_CONFIDENCE, config.YOLO_NMS_THRESHOLD)

    # default return values which are the image itself
    ret_x = 0
    ret_y = 0
    ret_width = W
    ret_height = H

    # ensure at least one detection exists
    if len(idxs) > 0:
        max_area = 0
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # apply a heuristic here
            # we assume that each image contains only 1 main car
            # so we only get the bounding box which has the largest area
            area = w * h
            if area > max_area:
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                max_area = area
                ret_x = x
                ret_y = y
                ret_width = w
                ret_height = h

    return ret_x, ret_y, ret_width, ret_height
