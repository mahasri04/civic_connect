from ultralytics import YOLO


model = YOLO("yolov8n-cls.pt")   


model.train(
    data="datasets",   
    epochs=20,       
    imgsz=224,        
    batch=16           
)


