# Desctiption
Detecting people in close proximity.


# How to run
1. Create venv
2. Activate venv
```bash
source avenv/bin/activate
```
3. Run the code with desired options
Save model:
```bash
python ./scripts/save_model.py
```

Run on video:
```bash
python main.py --test-file-path=./test_videos/actions2.mpg --save-video
```

Run using camera:
```bash
python main.py
```

# Settings
- slow  - bool flag to slow down execution
- activation_area - share of frame for person to take to be considered being close enough for trespassing
- iou_thr - IoU threshold
- conf_thr - confidence threshold

# Credits
I used YOLOv8n model for that. Made by [Ultralytics](https://docs.ultralytics.com/models/yolov8/).
