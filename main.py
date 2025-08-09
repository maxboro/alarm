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
from alarm.model import YoloTFLite1Class

COLOR_DETECTION = (255, 0, 0)
COLOR_TRESPASSING = (0, 0, 255)
SLOW = True

def draw_detections(frame, xyxy_array, imW, imH):
    for ind in range(len(xyxy_array)):
        x1, y1, x2, y2 = xyxy_array[ind]
        x1 = int(x1 * imW)
        y1 = int(y1 * imH)
        x2 = int(x2 * imW)
        y2 = int(y2 * imH)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DETECTION, 2)

def check_alarm_conditions(xywh_array):
    return False

def start_runtime_loop(args: argparse.Namespace):
    # Load the model
    yolo = YoloTFLite1Class(
        path=args.model,
        target_class_id=0, # person
        iou_thr=0.6,
        conf_thr=0.3
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
        if SLOW: time.sleep(2)
        ret, frame = video.read()
        if not ret:
            continue
        print("Got frame")
        results = yolo.track(frame)
        draw_detections(frame, results.xyxy_array, imW, imH)
        run_alarm = check_alarm_conditions(results.xywh_array)

        if run_alarm:
            print("Alarm")

        if args.display_video:
            cv2.imshow('frame', frame)
            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.save_video:
            out.write(frame)
    
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
