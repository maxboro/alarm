"""
Example
-------
```
 python main.py --test-file-path=./test_videos/actions2.mpg --save-video 
```
"""
import argparse
import cv2
from ultralytics import YOLO


COLOR_DETECTION = (255, 0, 0)
COLOR_TRESPASSING = (0, 0, 255)

def process_frame(frame, yolo):
    results = yolo.track(frame, stream=True)
    for result in results:
        # iterate over each box
        for box in result.boxes:
            object_cls = int(box.cls[0])
            class_name = result.names[object_cls]

            # filter by confidence
            if box.conf[0] > 0.4 and class_name == "person":
                [x1, y1, x2, y2] = [int(coord) for coord in box.xyxy[0]]

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DETECTION, 2)

                # put the class name and confidence on the image
                cv2.putText(
                    frame,
                    f'{class_name} {box.conf[0]:.2f}',
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, COLOR_DETECTION, 2
                )
    return False # temporaly

def start_runtime_loop(args: argparse.Namespace):
    # Load the model
    yolo = YOLO(args.model)

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
        run_alarm = process_frame(frame, yolo)

        if args.display_video:
            cv2.imshow('frame', frame)

        if args.save_video:
            out.write(frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ckeaning
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
        default="yolov8n.pt",
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
