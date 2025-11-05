from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")

# Export the model to OpenVINO format
model.export(format="openvino", half=True, dynamic=True, nms=True, imgsz=(480, 640))  # Export with FP16 precision and dynamic shapes
