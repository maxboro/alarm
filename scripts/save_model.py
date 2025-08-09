from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")
yolo.export(
    format="tflite",
    half=True,
    project=".",
    name="yolov8n_fp16.tflite"
)
