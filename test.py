import sys

# sys.path.append("./ultralytics")
sys.path.append("./fast_reid")
sys.path.append("./ultralytics")

from ultralytics import YOLO


model = YOLO()
model.track(
    source=r"C:\Users\Nadim\Videos\lab-2.mp4",
    persist=True,
    show=True,
    tracker="my_botsort.yaml",
)
