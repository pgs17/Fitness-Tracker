from ultralytics import YOLO
model=YOLO('yolov8m.pt')
model=YOLO('Big DAta.pt')

path=r"C:\Users\saran\Desktop\DATA SCIENCE AND ML\SIH 2023\test\images\russian-twist_1_frame189_jpg.rf.769a2f4ff4b54a5b9d0b21ee39306ed4.jpg"

model.predict(source=path,imgsz=640,show=True)