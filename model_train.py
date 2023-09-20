from ultralytics import YOLO

model=YOLO("yolov8l.yaml") # Give the model

model.train(data="data.yaml",epochs=320,imgsz=640)

# Run on colab after pip install ultralytics

