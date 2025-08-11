"""
Example
-------
```
 python main.py --test-file-path=./test_videos/actions2.mpg --save-video 
```
"""
import argparse
import cv2
import time
import yaml
import numpy as np
from alarm.model import YoloTFLite1Class, Detections

COLOR_DETECTION = (0, 255, 0)
COLOR_TRESPASSING = (0, 0, 255)

def draw_detections(frame: np.ndarray, results: Detections, imW: int, imH: int, trespassed_indxs: np.ndarray):
    for ind in range(len(results.xyxy_array)):
        x1, y1, x2, y2 = results.xyxy_array[ind]
        x1 = int(x1 * imW)
        y1 = int(y1 * imH)
        x2 = int(x2 * imW)
        y2 = int(y2 * imH)
        if trespassed_indxs[ind]:
            color = COLOR_TRESPASSING
            status = 'close'
        else:
            color = COLOR_DETECTION
            status = 'far'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f'Person {status} {results.conf_array[ind]:.2f}',
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, color, 2
        )

def read_settings() -> dict:
    with open("settings.yaml", "r") as settings_file:
        settings = yaml.safe_load(settings_file)
    return settings

def get_tresspassed(xywh_array: np.ndarray, settings: dict) -> np.ndarray:
    person_areas = xywh_array[:, 2] * xywh_array[:, 3]
    trespassed_indxs = person_areas > settings["activation_area"]
    return trespassed_indxs

def alarm_manager(frame: np.ndarray, trespassed_indxs: np.ndarray):
    if trespassed_indxs.sum() > 0:
        print("Alarm")
        cv2.putText(
            frame,
            'Alarm', 
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            COLOR_TRESPASSING, 2
        )

def start_runtime_loop(args: argparse.Namespace):
    settings = read_settings()
    # Load the model
    yolo = YoloTFLite1Class(
        path=args.model,
        target_class_id=0, # person
        iou_thr=settings["iou_thr"],
        conf_thr=settings["conf_thr"]
    )

    # init video capture
    file = args.test_file_path if args.test_file_path else 0
    video = cv2.VideoCapture(file)
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    if args.save_video:
        output_path = f"./test_output/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # (*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (imW, imH))
    else:
        out = None

    print("Video processing started")
    while True:
        ret, frame = video.read()
        if not ret:
            continue
        print("Got frame")
        results = yolo.track(frame)
        trespassed_indxs = get_tresspassed(results.xywh_array, settings)
        alarm_manager(frame, trespassed_indxs)
        draw_detections(frame, results, imW, imH, trespassed_indxs)

        if args.display_video:
            cv2.imshow('frame', frame)
            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save_video:
            out.write(frame)

        if settings["slow"]:
            time.sleep(2)

    # cleaning
    video.release()
    cv2.destroyAllWindows()
    if args.save_video:
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Alarm',
    )
    parser.add_argument(
        '--test-file-path', 
        type=str,
        default=None,
        help = 'Path to video file for testing.'
    )
    parser.add_argument(
        '--model', 
        type=str,
        default="./yolov8n_saved_model/yolov8n_float16.tflite",
        help = 'Object detection model name to use.'
    )
    parser.add_argument(
        '--save-video',
        help='Flag to save the video',
        action='store_true'
    )
    parser.add_argument(
        '--display-video',
        help='Flag to display the video',
        action='store_true'
    )
    args = parser.parse_args()
    start_runtime_loop(args)
