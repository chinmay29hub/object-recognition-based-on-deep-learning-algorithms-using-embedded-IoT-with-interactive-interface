import cv2
import numpy as np
import time

#loading yolo
net = cv2.dnn.readNet('yolov-tiny.weights', 'yolo-tiny.cfg') #yolo configration and re trained model weights

classes = []
with open("coco.names", "r") as f: # this are text of name files that are to be identifies
    classes = f.read().splitlines()

#layer_names = net.getLayerNames()
#outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#cap = cv2.VideoCapture('test.mp4') # input
cap=cv2.VideoCapture (0) #0 for 1st webcam font= cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    
    #showing info on the screen
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                #obj detected 
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h]) #show rectangle area
                confidences.append((float(confidence)))# how much is confidence
                class_ids.append(class_id)#name of obje that was detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = (confidences[i])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + str(round(confidence)), (x, y+30), font, 2, (255,255,255), 2)
    elapsed_time = time.time()  - starting_time
    fps=frame_id/elapsed_time
    cv2.putText (img, "FPS: "+str (round (fps, 2)), (10,50), font, 2, (0,0,0),1)

    # cv2_imshow(img)
    #cv2.resizeWindow("Resized_Window", 1000, 1000)
    #cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Resized_Window", 300, 700)
    #cv2.resizeWindow("Image", 700, 200)
    imS = cv2.resize(img, (1200, 800))
    cv2.imshow('Image', imS)
    key = cv2.waitKey (1) #wait 1ms the loop will start again and we will process the next frame
    if key ==27: #esc key stops the process ==
        break;

cap.release()
cv2.destroyAllWindows()
