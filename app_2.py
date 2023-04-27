import cv2
import numpy as np

# loading yolo
net = cv2.dnn.readNet('yolov-tiny.weights', 'yolo-tiny.cfg') # yolo configuration and retrained model weights

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# load image
img = cv2.imread('./object_img/8.jpg')

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes)>0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = (confidences[i])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 20)
        cv2.putText(img, label + " " + str(round(confidence, 20)), (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# resize the image to a smaller size
scale_percent = 50 # percent of original size
width = 500
height = 500
dim = (width, height)
img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# display the output image
cv2.imshow('Detected Image', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
