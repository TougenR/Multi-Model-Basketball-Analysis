import cv2
from copy import deepcopy

class TacticalViewConverter():
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300  # 300
        self.height = 161  # 161
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]


    def validate_keypoints(self, court_keypoints_list):
        court_keypoints_list = deepcopy(court_keypoints_list)
        for frame_idx, frame_keypoints in enumerate(court_keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            detected_indicies = [i for i, kp in enumerate(frame_keypoints) if kp[0]>0 and kp[1]>0]

            if len(detected_indicies) < 3:
                continue

            for i in detected_indicies:
                # skip (0,0) keypoints
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue












