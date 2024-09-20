import sys
import os
import torch
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
from threading import Thread
import requests  # For making HTTP requests

app = Flask(__name__)
CORS(app)

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the YOLO model (YOLOv5)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detections = []
        self.selected_box = None
        self.dragging = False
        self.start_pos = QPoint()

        # Setup GUI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Object Detection and Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.label = QLabel(self)
        self.label.setText("Upload an image to detect objects.")
        
        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        self.save_button = QPushButton("Save Annotations")
        self.save_button.clicked.connect(self.save_annotations)
        layout.addWidget(self.save_button)

        self.label_dropdown = QComboBox()
        self.label_dropdown.addItems(["person", "car", "truck", "bicycle"])
        self.label_dropdown.currentTextChanged.connect(self.change_label)
        layout.addWidget(self.label_dropdown)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            print(f"Selected file: {file_name}")
            self.detect_objects(file_name)

    def detect_objects(self, image_path):
        # Send the image to the Flask server for detection
        try:
            with open(image_path, 'rb') as f:
                response = requests.post('http://127.0.0.1:5000/detect', files={'image': f})
                if response.status_code == 200:
                    data = response.json()
                    self.detections = data.get('detections', [])
                    self.update_image_display(image_path)
                else:
                    print(f"Error: {response.json().get('error')}")
        except Exception as e:
            print(f"An error occurred while detecting objects: {e}")

    def update_image_display(self, image_path):
        # Load the annotated image path
        annotated_image_path = 'uploads/annotated_' + os.path.basename(image_path)

        # Read the annotated image
        annotated_image = cv2.imread(annotated_image_path)
        display_image = annotated_image.copy()

        for det in self.detections:
            rect = det['rect']
            label = det['label']
            cv2.rectangle(display_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.putText(display_image, label, (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        height, width, _ = display_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def change_label(self, label):
        if self.selected_box:
            self.selected_box['label'] = label
            self.update_image_display()

    def save_annotations(self):
        print("Annotations saved:", self.detections)

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
    # Start the Flask server in a separate thread
    Thread(target=run_flask).start()

    # Start the PyQt5 application
    qt_app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()

    sys.exit(qt_app.exec_())