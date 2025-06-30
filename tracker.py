
import numpy as np
from supervision import ByteTrack
from supervision.detection.core import Detections

class Tracker:
    def __init__(self, reid_model):
        self.reid = reid_model
        self.tracker = ByteTrack()
        self.frame_count = 0

    def update(self, frame, detections):
        self.frame_count += 1
        xyxy, confs, cls_ids = [], [], []
        for x1, y1, x2, y2, conf in detections:
            xyxy.append([x1, y1, x2, y2])
            confs.append(conf)
            cls_ids.append(0)

        if xyxy:
            xyxy_arr = np.array(xyxy, dtype=np.float32)
            conf_arr = np.array(confs, dtype=np.float32)
            class_arr = np.array(cls_ids, dtype=int)
        else:
            xyxy_arr = np.zeros((0,4), dtype=np.float32)
            conf_arr = np.zeros((0,), dtype=np.float32)
            class_arr = np.zeros((0,), dtype=int)

        dets = Detections(xyxy=xyxy_arr, confidence=conf_arr, class_id=class_arr)
        tracked = self.tracker.update_with_detections(detections=dets)

        output = []
        for (x1, y1, x2, y2), tid, conf in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            feat = self.reid.extract_feature(crop)
            mid = self.reid.match(feat, frame_id=self.frame_count)
            final_id = mid if mid is not None else int(tid)
            self.reid.update_memory(final_id, feat, self.frame_count)
            output.append((final_id, [int(x1), int(y1), int(x2), int(y2), float(conf)]))
        return output
