Object Detection Project in Google Colab

Step 1: Install Required Libraries

PyTorch and torchvision are used for model loading and processing

Matplotlib is used for visualization

OpenCV and PIL are used for image handling

!pip install torch torchvision matplotlib opencv-python



Step 2: Import Required Libraries

import torch  # Core PyTorch library for tensor operations and model inference
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # Pre-trained Faster R-CNN model
from torchvision.transforms import functional as F  # Image transformation utilities
from torchvision.utils import draw_bounding_boxes  # For drawing bounding boxes
from torchvision.transforms import ToPILImage  # Convert tensor to PIL image
import matplotlib.pyplot as plt  # Visualization library
from PIL import Image  # Image processing library
import cv2  # OpenCV for image handling
import numpy as np  # Numerical computations



Step 3: Load Pre-Trained Model

Faster R-CNN pre-trained on COCO dataset is loaded

The model is set to evaluation mode to avoid training-related computations

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode



Step 4: Upload and Test an Image

Google Colab provides a file upload feature to easily upload images

from google.colab import files  # Colab-specific file upload library
uploaded = files.upload()  # Upload an image file
image_path = list(uploaded.keys())[0]  # Get the uploaded file name
image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB

Convert image to PyTorch tensor

image_tensor = F.to_tensor(image)  # Transform image to tensor



Step 5: Perform Object Detection

Perform inference using the pre-trained model

with torch.no_grad():  # Disable gradient computation for inference
predictions = model([image_tensor])

Extract bounding boxes, labels, and scores from the model's output

boxes = predictions[0]['boxes']  # Bounding box coordinates
labels = predictions[0]['labels']  # Detected class labels
scores = predictions[0]['scores']  # Confidence scores



Step 6: Visualize the Results

Filter detections with confidence scores above a threshold

threshold = 0.5  # Confidence threshold for filtering
keep = scores > threshold  # Boolean mask for valid detections

Draw bounding boxes with labels on the image

image_with_boxes = draw_bounding_boxes(
(image_tensor * 255).byte(),  # Convert tensor to byte scale for visualization
boxes[keep],  # Filtered bounding boxes
colors="red",  # Bounding box color
labels=[f"{label.item()}" for label in labels[keep]],  # Class labels for boxes
width=3  # Line width of bounding boxes
)

Convert tensor back to PIL image for display

plt.figure(figsize=(12, 8))  # Set figure size
plt.imshow(ToPILImage()(image_with_boxes))  # Display the image with boxes
plt.axis("off")  # Remove axes for cleaner display
plt.show()  # Show the visualization



Step 7: Save the Results

Save the processed image to the Colab environment

output_path = "detected_image.jpg"  # Output file name
image_with_boxes_pil = ToPILImage()(image_with_boxes)  # Convert to PIL image
image_with_boxes_pil.save(output_path)  # Save image locally
print(f"Detected image saved at {output_path}")  # Confirm save location

Allow user to download the result
