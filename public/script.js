let selectedBox = null;
let isDragging = false;
let isResizing = false;
let isDrawing = false;
let startX, startY;
let newBox = null;
const boxes = [];

// Set up event listeners for the upload button
document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const statusMessage = document.getElementById('statusMessage');
    const resultImage = document.getElementById('resultImage');

    if (!fileInput.files.length) {
        statusMessage.innerText = "Please select an image file.";
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('http://127.0.0.1:5000/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            statusMessage.innerText = `Error: ${error.error}`;
            return;
        }

        const data = await response.json();
        statusMessage.innerText = "Detection successful!";
        resultImage.src = 'http://127.0.0.1:5000/' + data.annotated_image_path;
        resultImage.style.display = "block";

        const canvas = document.getElementById('resultCanvas');
        canvas.width = resultImage.naturalWidth;
        canvas.height = resultImage.naturalHeight;

        // Clear previous boxes before adding new detections
        boxes.length = 0; // Clear the boxes array
        data.detections.forEach(detection => {
            boxes.push({
                rect: detection.rect,
                label: detection.label // Ensure labels are stored correctly
            });
        });

        drawBoxes(boxes);
    } catch (error) {
        console.error('Error:', error);
        statusMessage.innerText = "An error occurred while processing the image.";
    }
});

// Draw the bounding boxes on the canvas
function drawBoxes(boxes) {
    const canvas = document.getElementById('resultCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    boxes.forEach(box => {
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.rect[0], box.rect[1], box.rect[2] - box.rect[0], box.rect[3] - box.rect[1]);

        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(box.label, box.rect[0], box.rect[1] - 5);
    });
}

// Canvas mouse events for dragging and resizing boxes
const canvas = document.getElementById('resultCanvas');
canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;

    selectedBox = boxes.find(box => isPointInBox(startX, startY, box));
    if (selectedBox) {
        const isNearEdge = checkNearEdge(startX, startY, selectedBox);
        if (isNearEdge) {
            isResizing = true;
        } else {
            isDragging = true;
        }
    } else {
        isDrawing = true;
        newBox = { rect: [startX, startY, startX, startY], label: document.getElementById('classLabelDropdown').value === 'custom' ? document.getElementById('customLabelInput').value : document.getElementById('classLabelDropdown').value };
    }
});

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging && selectedBox) {
        const dx = mouseX - startX;
        const dy = mouseY - startY;

        selectedBox.rect[0] += dx;
        selectedBox.rect[1] += dy;
        selectedBox.rect[2] += dx;
        selectedBox.rect[3] += dy;

        drawBoxes(boxes);
        startX = mouseX;
        startY = mouseY;
    }

    if (isResizing && selectedBox) {
        selectedBox.rect[2] = mouseX;
        selectedBox.rect[3] = mouseY;
        drawBoxes(boxes);
    }

    if (isDrawing) {
        newBox.rect[2] = mouseX;
        newBox.rect[3] = mouseY;
        drawBoxes([...boxes, newBox]);
    }
});

canvas.addEventListener('mouseup', () => {
    if (isDragging) {
        isDragging = false;
        selectedBox = null;
    } else if (isDrawing) {
        boxes.push(newBox);
        isDrawing = false;
        newBox = null;
    } else if (isResizing) {
        isResizing = false;
        selectedBox = null;
    }
});

document.getElementById('classLabelDropdown').addEventListener('change', (e) => {
    if (e.target.value === 'custom') {
        document.getElementById('customLabelInput').style.display = 'inline';
    } else {
        document.getElementById('customLabelInput').style.display = 'none';
        if (selectedBox) {
            selectedBox.label = e.target.value;
            drawBoxes(boxes);
        }
    }
});

// Helper function to check if a point is inside a box
function isPointInBox(x, y, box) {
    return x > box.rect[0] && x < box.rect[2] && y > box.rect[1] && y < box.rect[3];
}

// Helper function to check if the point is near the edge of a box (for resizing)
function checkNearEdge(x, y, box) {
    const edgeThreshold = 10;
    const nearRightEdge = Math.abs(x - box.rect[2]) < edgeThreshold;
    const nearBottomEdge = Math.abs(y - box.rect[3]) < edgeThreshold;
    return nearRightEdge || nearBottomEdge;
}
