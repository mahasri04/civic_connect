from ultralytics import YOLO
model = YOLO("runs/classify/train/weights/best.pt")

results = model.predict("test.jpg")

for r in results:
    probs = r.probs  
    names = model.names
    for i, p in enumerate(probs):
        print(f"{names[i]}: {float(p):.2f}")
