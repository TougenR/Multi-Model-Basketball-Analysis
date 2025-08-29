from utils import save_stub, read_stub
from ultralytics import YOLO
import supervision as sv
import pandas as pd
import numpy as np
import sys
sys.path.append("../")


class BallTracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 16
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections.extend(batch_detections)

        return detections

    def objects_track(self, frames, read_from_stub=False, stub_path=None):
        # read from lastest checkpoint
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []
        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v: k for k, v in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0

            for frames_detection in detection_supervision:
                bbox = frames_detection[0].tolist()
                confidence = frames_detection[2]
                cls_id = frames_detection[3]

                if cls_id == cls_name_inv["Ball"]: # Ball
                    if confidence > max_confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        maximum_allowed_distance = 15
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(
                1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}}
                          for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
        """
        Before interpolate:
           x1   y1   x2   y2
        0  1.0  5.0  2.0  3.0
        1  NaN  NaN  NaN  NaN
        2  4.0  4.0  7.0  5.0
        3  5.0  NaN  8.0  6.0
        ----------------------
        After interpolate:
            x1   y1   x2   y2
        0  1.0  5.0  2.0  3.0
        1  2.5  4.5  4.5  4.0
        2  4.0  4.0  7.0  5.0
        3  5.0  4.0  8.0  6.0 
        
        y tuong: su dung phuong phap noi suy tu cac frame lien ke de tinh trung binh vi tri hien tai cua frame trong
        """








