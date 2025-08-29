import cv2


class TacticalViewDrawer():
    def __init__(self):
        # position of the top left corner of the court image
        self.start_x = 450
        self.start_y = 20

    def draw(self,
             video_frames,
             court_image_path,
             width,
             height,
             court_keypoints_list
             ): 
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            output_frames = frame.copy()
            
            x1 = self.start_x
            y1 = self.start_y
            x2 = x1 + width
            y2 = y1 + height

            alpha = 0.6 # transparency
            overlay = output_frames[y1:y2,x1:x2].copy()
            cv2.addWeighted(court_image,
                            alpha,
                            overlay,
                            1-alpha,
                            0,
                            output_frames[y1:y2,x1:x2])
            
            for keypoint_idx, keypoint in enumerate(court_keypoints_list):
                x, y = keypoint
                x += self.start_x
                y += self.start_y
                center = (int(x), int(y))
                cv2.circle(output_frames,
                           center,
                           radius=5,
                           color=(0,0,255),
                           thickness=-1)

                cv2.putText(output_frames,
                            str(keypoint_idx),
                            center,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            2)


            output_video_frames.append(output_frames)

        return output_video_frames

        
