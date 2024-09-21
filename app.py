import sys
import os
import torch
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import Thread
import requests  # For making HTTP requests

app = Flask(__name__)
CORS(app)

@app.route('/detect', methods=['POST'])
def detect_objects_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']

    # Ensure uploads directory exists
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    image_path = os.path.join(upload_dir, file.filename)
    file.save(image_path)

    # Load the YOLO model (YOLOv5)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    image = cv2.imread(image_path)
    results = model(image)

    detections = []
    for det in results.xyxy[0].numpy():
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        detections.append({
            'rect': [int(x1), int(y1), int(x2), int(y2)],
            'label': label
        })

    # Optional: Save an annotated image
    annotated_image_path = os.path.join(upload_dir, 'annotated_' + file.filename)
    for det in detections:
        rect = det['rect']
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.putText(image, det['label'], (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(annotated_image_path, image)

    os.remove(image_path)  # Clean up the saved image
    return jsonify({'detections': detections, 'annotated_image_path': 'uploads/annotated_' + file.filename})

@app.route('/uploads/<path:filename>', methods=['GET'])
def send_uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/<path:path>')
def send_js(path):
    return send_from_directory('public', path)

@app.route('/')
def serve():
    return send_from_directory('public', 'index.html')

def run_flask():
    app.run(port=5000)

if __name__ == "__main__":
    # Start the Flask server
    run_flask()
