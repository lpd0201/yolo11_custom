

from ultralytics import YOLO
import torch

model = YOLO("C:/Users/DUONG/Desktop/Paper_Q3_YOLO/yolo_custom.yaml")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    rs = model.train(
        epochs = 2,
        data = "C:/Users/DUONG/Desktop/Paper_Q3_YOLO/VisDrone2019.yaml",
        imgsz = 640,
        batch=4,         
        workers=2,         
        pretrained=False,
        device = 0
    )
