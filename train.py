from ultralytics import YOLO
model = YOLO('yolov8s.pt')

model.train(data='dataset.yaml' , epochs=20)