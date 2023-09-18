from ultralytics import YOLO

model=YOLO("") # Give the model

model.train(data="data.yaml",epochs=300,imgsz=640)