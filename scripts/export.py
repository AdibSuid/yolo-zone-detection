from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Export the model to OpenVINO format
model.export(format="openvino", half=True, dynamic=True, nms=True, imgsz=(1080, 1920))  # Export with FP16 precision and dynamic shapes
