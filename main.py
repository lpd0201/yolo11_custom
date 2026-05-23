

from ultralytics import YOLO
import torch

model = YOLO("C:/Users/DUONG/Desktop/Paper_Q3_YOLO/ultralytics/cfg/models/11/adpanet_yolo.yaml")

if __name__ == "__main__":
    rs = model.train(
        epochs = 2,
        data = "C:/Users/DUONG/Desktop/Paper_Q3_YOLO/codeandmodule/VisDrone2019.yaml",
        imgsz = 640,
        batch=4,         
        workers=2,         
        pretrained=False,
        device = 0,
        amp = False
    )

