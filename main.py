

from ultralytics import YOLO
import torch

model = YOLO("C:/Users/DUONG/Desktop/Paper_Q3_YOLO/ultralytics/cfg/models/11/adpanet_yolo.yaml")

if __name__ == "__main__":
    rs = model.train(
        data = "C:/Users/DUONG/Desktop/Paper_Q3_YOLO/codeandmodule/VisDrone2019.yaml",
        imgsz = 640,
        epochs=2,
        batch=4,                  
        device = 0,
        momentum=0.937,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        amp=False
    )



