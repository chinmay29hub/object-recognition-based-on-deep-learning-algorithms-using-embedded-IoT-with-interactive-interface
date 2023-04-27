import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loop through images 1 to 20
for i in range(1, 21):
    # Loading image
    img = cv2.imread(f"object_img/{i}.jpg")
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle co-ordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])  # Put all rectangle areas
                confidences.append(float(confidence))  # How confidence was that object detected and show that percentage
                class_ids.append(class_id)  # Name of the object that was detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    # Create ground truth file
    with open("prediction.txt", "a") as f:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                f.write(str(x) + "," + str(y) + "," + str(x + w) + "," + str(
                    y + h) + "\n")

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
