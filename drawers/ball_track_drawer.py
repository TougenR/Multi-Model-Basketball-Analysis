from .util import draw_triangle


class BallTrackDrawer():
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frame, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frame):
            output_frame = frame.copy() # tao ban sao de tranh ghi de len video goc
            ball_dict = tracks[frame_num]

            for _, ball in ball_dict.items():
                bbox = ball["bbox"]
                if bbox is None:
                    continue
                output_frame = draw_triangle(
                    output_frame, bbox, self.ball_pointer_color)

            output_video_frames.append(output_frame)

        return output_video_frames
