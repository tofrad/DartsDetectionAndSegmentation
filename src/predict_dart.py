from torchvision import transforms, models
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available! Using GPU: {gpu_name}")

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the index of the currently selected device
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device index: {current_device}")

    # You can explicitly set the device to use
    device = torch.device("cuda")
else:
    print("CUDA is NOT available. PyTorch will run on the CPU.")
    device = torch.device("cpu")

print(torch.__version__)
print(device)

# Label list
CLASS_NAMES = [
    "BG", "arrow", "board"
    ]
#
# CLASS_COLORS = {
#     "arrow_1": (0, 165, 255),    # Orange
#     "arrow_2": (255, 0, 0),      # Blau
#     "arrow_3": (0, 255, 0),      # Grün
#     "board": (255, 255, 0),      # Cyan
#     # ...
# }

# Get the number of classes (including background)
num_classes = len(CLASS_NAMES) # background + your classes

#cannot load whole pytorch models at the moment, recursion bug inside pytorch  api
# def get_model(num_classes):
#     model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
#
#     # Box Predictor
#     in_features_box = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
#
#     # Mask Predictor
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
#
#     return model

# Load Mask R-CNN model with correct number of classes
model = models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

# Load your trained weights
model.load_state_dict(torch.load("only_model_epoch_8.pth"))

#run on gpu, not usable atm, needs bugfix
#model.to(device)

#set model to inference mode
model.eval()

# Load image with OpenCV and convert to RGB
img_path = r"real_test_2.jpg"  # Change this path
image_bgr = cv2.imread(img_path)

#image_bgr = cv2.resize(original_image_bgr, (1024, 768))

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_pil = Image.fromarray(image_rgb)

# Transform image to tensor and add batch dimension
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0)

# Inference
with torch.no_grad():
    predictions = model(image_tensor)

# Extract masks, boxes, labels, and scores
masks = predictions[0]['masks']       # [N, 1, H, W]
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

threshold = 0.9  # Confidence threshold

# Use overlay for blending masks over image
overlay = image_bgr.copy()

for i in range(len(masks)):
    if scores[i] > threshold:
        # Convert mask to uint8 numpy array (H,W)
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        mask_bool = mask > 127  # binary mask for indexing
        box = boxes[i].cpu().numpy().astype(int)
        #print(labels[i])
        class_name = CLASS_NAMES[labels[i]]
        score = scores[i].item()

        # # Generate random color (BGR)
        # color = CLASS_COLORS.get(class_name,
        #                          np.random.randint(0, 255, 3).tolist())

        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

        # Create colored mask with the random color
        colored_mask = np.zeros_like(image_bgr, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask_bool * color[c]

        # Alpha blend the colored mask onto the overlay
        alpha = 0.5
        overlay = np.where(mask_bool[:, :, None],
                           ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8),
                           overlay)

        # Draw bounding box and label text on overlay
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA)

# Show the result using matplotlib (convert BGR -> RGB)
result_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
result_path = r"real_test_2_detected.jpg"
cv2.imwrite(result_path, overlay)

plt.figure(figsize=(12, 8))
plt.imshow(result_rgb)
plt.axis('off')
plt.show()