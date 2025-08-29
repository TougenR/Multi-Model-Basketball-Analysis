from ultralytics import YOLO
from utils import read_stub, save_stub
import sys
sys.path.append("../")


class CourtKeypointDetector():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def key_court_keypoints(self, frame, read_from_stub=False, stub_path=None):
        
        court_keypoints_detection = read_stub(read_from_stub, stub_path)
        if court_keypoints_detection is not None:
            if len(court_keypoints_detection) == len(frame):
                return court_keypoints_detection

        batch_size = 20
        court_keypoints_detection = []
        for i in range(0, len(frame), batch_size):
            batch_frame = frame[i:i+batch_size]
            detection_batch = self.model.predict(batch_frame, conf=0.5)
            for detection in detection_batch:
                court_keypoints_detection.append(detection.keypoints)
        
        save_stub(stub_path, court_keypoints_detection)
        return court_keypoints_detection

            
            


