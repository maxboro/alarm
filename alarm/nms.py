import numpy as np

def nms_xyxy_single_class(boxes, scores, iou_thres, conf_thres, topk=300):
    """
    boxes: (N, 4) array of [x1, y1, x2, y2] in pixels
    scores: (N,) array of confidence scores
    Returns: indices of kept boxes
    """
    # 1) Confidence filter
    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0,), dtype=int)

    boxes = boxes[mask]
    scores = scores[mask]
    idxs = np.nonzero(mask)[0]  # mapping to original indices

    # 2) Top-K filter
    if topk is not None and len(scores) > topk:
        order = scores.argsort()[::-1][:topk]
        boxes = boxes[order]
        scores = scores[order]
        idxs = idxs[order]

    # 3) NMS loop
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_thres]

    return idxs[keep]
