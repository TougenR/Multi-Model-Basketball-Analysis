from utils import save_stub, read_stub, get_device, setup_model_for_gpu, get_optimal_batch_size, optimize_inference_settings
from ultralytics import YOLO
import supervision as sv
import torch
import sys
sys.path.append("../")


class PlayerTracker():
    def __init__(self, model_path):
        # Initialize GPU settings
        optimize_inference_settings()
        self.device = get_device()

        # Load and setup model for GPU
        self.model = YOLO(model_path)
        self.model, self.device = setup_model_for_gpu(self.model, self.device)

        # Optimize batch size based on device
        self.batch_size = get_optimal_batch_size(self.device.type, 'medium')

        # Initialize tracker
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]

            # Use GPU device for inference
            batch_detections = self.model.predict(
                batch_frames,
                conf=0.5,
                device=self.device,
                verbose=False  # Reduce logging for better performance
            )
            detections.extend(batch_detections)

        return detections

    def objects_track(self, frames, read_from_stub=False, stub_path=None):
        # doc tu checkpoint gan nhat => tiet kiem thoi gian thuc thi code
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v: k for k, v in cls_name.items()}

            # convert from YOLO format to supervision format so we can use self.tracker
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_track = self.tracker.update_with_detections(
                detection_supervision)

            # tao mot dict rong de chua cac bbox trong frame_num do
            tracks.append({})

            for frames_detection in detection_with_track:
                bbox = frames_detection[0].tolist()
                cls_id = frames_detection[3]
                track_id = int(frames_detection[4])

                if cls_id == cls_name_inv["Player"]:
                    tracks[frame_num][track_id] = {"bbox": bbox}
        """
            the result would be like this:
            tracks = [
                # Frame 0 (index 0)
                {
                    101: {"bbox": [10, 20, 50, 80]},  # key=101, value={"bbox": bbox}
                    102: {"bbox": [60, 30, 100, 90]}  # key=102, value={"bbox": bbox}
                },
                # Frame 1 (index 1)
                {
                    101: {"bbox": [12, 22, 52, 82]},  # key=101, value={"bbox": bbox}
                    103: {"bbox": [70, 40, 110, 100]} # key=103, value={"bbox": bbox}
                }
            ]
        """
        save_stub(stub_path, tracks)
        return tracks
