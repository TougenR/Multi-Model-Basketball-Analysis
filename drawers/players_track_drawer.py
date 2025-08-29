from .util import draw_ellipse, draw_triangle


class PlayersTrackDrawer():
    def __init__(self,
                 team_1_color=[255, 245, 235],
                 team_2_color=[179, 0, 0]
                 ):

        # use default player team color when the model miss-detect
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frame, tracks, player_assignment, ball_acquisition):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frame):
            output_frame = frame.copy()  # make a copy frame to not to override the orginal frame
            player_dict = tracks[frame_num]
            player_team_assignment_for_frame = player_assignment[frame_num]
            player_id_has_ball = ball_acquisition[frame_num]
            for track_id, player in player_dict.items():
                team_id = player_team_assignment_for_frame.get(
                    track_id, self.default_player_team_id)
                # draw player team color
                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                # draw a triangle on player that has ball
                if track_id == player_id_has_ball:
                    output_frame = draw_triangle(
                        output_frame, player["bbox"], (0, 0, 255))

                output_frame = draw_ellipse(
                    output_frame, player["bbox"], color, track_id)

            output_video_frames.append(output_frame)

        return output_video_frames
