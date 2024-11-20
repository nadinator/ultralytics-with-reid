#! /home/nboualwa/miniconda3/envs/ultrafast/bin/python
import sys

sys.path.append("./fast_reid")
sys.path.append("./ultralytics")

from ultralytics import YOLO


model = YOLO(
    "./best_models/train/oct31_v8x--run4--optimizerAdamW_dropout0.4_lrf0.0001_lr01e-05/weights/best.pt",
)

if __name__ == "__main__":
    model.track(
        source="rtsp://192.168.1.3/live",
        persist=True,
        show=True,
        tracker="my_botsort.yaml",
    )
