from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import random

app = Flask(__name__)

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

score = 0
current_object = []

def detect_objects(frame):
    global current_object
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
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
            if confidence > 0.1:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = np.random.uniform(0, 255, size=(3,))
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            current_object.append(label)

    return frame

def gen_frames(): 
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    global score, current_object
    score = 0
    current_object = []
    return render_template('game.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_object', methods=['POST'])
def detect_object():
    global score, current_object
    if request.method == 'POST':
        object_name = request.form['object_name']
        if object_name in current_object:
            score += 10
            current_object = []
    return str(score)

if __name__ == '__main__':
    app.run(debug=True, port=4000)
