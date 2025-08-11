import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple
from dataclasses import dataclass
from .nms import nms_xyxy_single_class

@dataclass
class Detections:
    xywh_array: np.ndarray
    xyxy_array: np.ndarray
    conf_array: np.ndarray

class YoloTFLite1Class:
    def __init__(self, path, target_class_id, iou_thr, conf_thr):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.target_class_id = target_class_id
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr

    def _get_output(self, frame):
        """Raw output from tflite model, (1, 84, 8400) shaped vector."""
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    @staticmethod
    def _get_per_candidate_row(data: np.ndarray) -> np.ndarray:
        return data[0].T

    @staticmethod
    def _xywh2xyxy(xywh_array: np.ndarray) -> np.ndarray:
        x, y, w, h = xywh_array.T
        x1 = x - w / 2
        y1 =  y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        xyxy_array = np.stack([x1, y1, x2, y2]).T
        xyxy_array = np.clip(xyxy_array, 0, 1)
        return xyxy_array

    def _split_to_boxes_and_confidence(self, data: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        xywh_array = data[:, :4]
        conf_array = data[:, 4 + self.target_class_id]
        return xywh_array, conf_array

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        inp = (cv2.resize(frame, (640, 640)) / 255.0).astype(np.float32)
        inp = np.expand_dims(inp, 0)
        return inp
    
    def track(self, frame: np.ndarray)-> Detections:
        inp = self._preprocess_frame(frame)
        output_data = self._get_output(inp)
        output_data = self._get_per_candidate_row(output_data)
        xywh_array, conf_array = self._split_to_boxes_and_confidence(output_data)
        xyxy_array = self._xywh2xyxy(xywh_array)
        detected_idx = nms_xyxy_single_class(
            xyxy_array,
            conf_array,
            iou_thres=self.iou_thr,
            conf_thres=self.conf_thr
        )
        results = Detections(
            xywh_array[detected_idx],
            xyxy_array[detected_idx],
            conf_array[detected_idx]
        )
        return results
