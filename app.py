import sys
import torch
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QPoint


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the YOLO model (YOLOv5)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.image_path = "D:\LabelDetector\download (1).jpeg"  # Change this to your image path
        self.detections = []
        self.selected_box = None
        self.dragging = False
        self.start_pos = QPoint()
        self.current_label = []  # Default class label

        # Load image and detect objects
        self.image = cv2.imread(self.image_path)
        self.detect_objects()

        # Setup GUI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Object Detection and Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.label = QLabel(self)
        self.update_image_display()

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        self.save_button = QPushButton("Save Annotations")
        self.save_button.clicked.connect(self.save_annotations)
        layout.addWidget(self.save_button)

        self.label_dropdown = QComboBox()
        self.label_dropdown.addItems(["person", "car", "truck", "bicycle"])  # Add more classes as needed
        self.label_dropdown.currentTextChanged.connect(self.change_label)
        layout.addWidget(self.label_dropdown)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def detect_objects(self):
    # Run inference on the image using the loaded YOLO model
        results = self.model(self.image)

        # Extract bounding boxes, confidence scores, and class labels
        for det in results.xyxy[0].numpy():
            x1, y1, x2, y2, conf, cls = det
            label = self.model.names[int(cls)]  # Get the class name from the model's class index
            
            # Create a bounding box and set the default label to the model's prediction
            self.detections.append({
                'rect': QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 
                'label': label  # Set default label to model's prediction
            })


    def update_image_display(self):
    # Convert the image to QPixmap for display in PyQt5
        display_image = self.image.copy()
        
        # Loop through detected bounding boxes
        for det in self.detections:
            self.current_label.append(det['label'])
            rect = det['rect']
            label = det['label']

            # Draw bounding box
            cv2.rectangle(display_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

            # Text settings
            font_scale = 0.4
            font_thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

            # Position of the label
            text_x = rect.left()
            text_y = rect.top() - 5 if rect.top() - 5 > 10 else rect.top() + 20
            
            # Create a background rectangle for the label
            cv2.rectangle(
                display_image, 
                (text_x, text_y - text_size[1] - 5), 
                (text_x + text_size[0] + 10, text_y + 5), 
                (0, 0, 0), 
                cv2.FILLED
            )

            # Draw the label text
            cv2.putText(
                display_image, 
                label, 
                (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255),  # Black text for contrast
                font_thickness, 
                lineType=cv2.LINE_AA
            )

        # Convert the OpenCV image to QImage and then to QPixmap
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.selected_box = self.get_selected_box(self.start_pos)
            
            # Check if a bounding box was clicked
            if self.selected_box:
                self.current_label = self.selected_box['label']
                self.label_dropdown.setCurrentText(self.current_label)  # Update the dropdown to reflect the current label
                self.dragging = True
            else:
                self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_box:
            delta = event.pos() - self.start_pos
            self.selected_box['rect'].moveTopLeft(self.selected_box['rect'].topLeft() + delta)
            self.start_pos = event.pos()
            self.update_image_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.selected_box = None

    def get_selected_box(self, pos):
        # Convert the mouse position into the image coordinates
        for det in self.detections:
            rect = det['rect']
            # Check if the mouse position is within any bounding box
            if rect.contains(self.label.mapFromGlobal(pos)):
                return det
        return None


    def change_label(self, label):
        # self.current_label = label
        if self.selected_box:
            self.selected_box['label'] = label
            self.update_image_display()

    def save_annotations(self):
        # Save modified annotations to a file or format of your choice
        print("Annotations saved:", self.detections)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
