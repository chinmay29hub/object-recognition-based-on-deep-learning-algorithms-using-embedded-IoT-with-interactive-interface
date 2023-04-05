from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

def detect_objects(frame):
    if frame is None:
        return None, []

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

    objects_detected = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = np.random.uniform(0, 255, size=(3,))
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y+20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            objects_detected.append(label)

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return frame, objects_detected


def gen_frames(): 
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame is not None:
            frame, objects_detected = detect_objects(frame)
            if frame is None:
                print('frame is None')
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('game_2.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_image', methods=['POST'])
def save_image():
    img_data = request.json['img_data']
    img_data = img_data.replace('data:image/jpeg;base64,', '')  # remove header info

    # convert base64 string to PIL Image object
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))

    # save image to disk
    img.save('image.jpg')

    return 'Image saved to disk!'

@app.route('/detect_objects', methods=['POST'])
def detect_objects_route():
    img_data = request.json['img_data']
    img_data = img_data.replace('data:image/jpeg;base64,', '')
    # with open("./image.jpg", "rb") as image_file:
    #     img_data = base64.b64encode(image_file.read())
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, objects_detected = detect_objects(img_np)
    temp = set(objects_detected)
    temp2 = list(temp)
    print(temp)

    return {'objects_detected': temp2}


if __name__ == '__main__':
    app.run(debug=True)
