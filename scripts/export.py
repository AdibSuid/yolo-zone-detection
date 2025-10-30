from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")

# Export the model to OpenVINO format
model.export(format="openvino", half=True)  # Export with FP16 precision
