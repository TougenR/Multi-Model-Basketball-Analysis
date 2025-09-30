import numpy as np
import tactical_view_converter
from tracker import PlayerTracker, BallTracker
from utils import read_video, save_video
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import (
    TacticalViewConverter,
    TacticalViewConverterOptimized,
    CourtModel
)
from drawers import (
    PlayersTrackDrawer,
    BallTrackDrawer,
    TeamBallControlDrawer,
    PassInterceptionTableDrawer,
    CourtKeypointsDrawer,
    TacticalViewDrawer
)


def main():
    # read video
    video_frames = read_video("./input_videos/video_1.mp4")

    # Initialize models
    player_tracker = PlayerTracker("./models/player_detector.pt")
    ball_tracker = BallTracker("./models/ball_detector_model.pt")
    court_keypoint_detector = CourtKeypointDetector(
        "./models/court_keypoint_detector.pt")

    player_tracks = player_tracker.objects_track(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/stub_player_tracks.pkl"
    )

    ball_tracks = ball_tracker.objects_track(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/stub_ball_tracks.pkl"
    )
    
    # Remove wrong ball positions
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Initialize drawers
    players_tracks_drawer = PlayersTrackDrawer()
    ball_tracks_drawer = BallTrackDrawer()
    # default table_position='top_left', transparency=0.8
    team_ball_control_drawer = TeamBallControlDrawer(transparency=0.6)
    pass_and_interception_drawer = PassInterceptionTableDrawer(
        table_position="top_right", transparency=0.6)
    court_keypoints_drawer = CourtKeypointsDrawer()

    # Initialize tactical view drawer with perspective mode and top positioning
    tactical_view_drawer = TacticalViewDrawer(
        perspective_mode=True,
        transparency=0.7  # Higher transparency for better overlay visibility
    )

    # Player team assigner
    team_assigner = TeamAssigner()
    players_assignment = team_assigner.get_player_team_across_frame(
        video_frames,
        player_tracks,
        read_from_stub=True,
        stub_path="stubs/stub_player_assignment.pkl"
    )

    # Team assignments loaded successfully

    # Ball Acquisition
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(
        player_tracks, ball_tracks)

    # pass and interception detector
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(
        ball_acquisition, players_assignment)
    interceptions = pass_and_interception_detector.detect_interception(
        ball_acquisition, players_assignment)

    # court keypoints detector - use cached stubs for consistency
    print("Loading court keypoints from cached stubs...")
    court_keypoint = court_keypoint_detector.key_court_keypoints(video_frames,
                                                                 read_from_stub=True,
                                                                 stub_path="stubs/stub_court_keypoints.pkl"
                                                                 )
    print(f"Loaded {len(court_keypoint)} frames of court keypoints")

    # Initialize tactical view converter with original approach (hardcoded template keypoints)
    # Use hardcoded court template for consistent layout, detected keypoints only for transformation
    court = CourtModel(width=400, height=200, actual_width_meters=30)
    tactical_view_converter = TacticalViewConverterOptimized(court_model=court)

    # Skip validation to use raw detected keypoints for transformation matrix computation
    # Template keypoints are hardcoded in court model for consistent court layout
    print("Using hardcoded court template for tactical view with detected keypoints for transformation")

    # Draw Object
    output_video_frames = ball_tracks_drawer.draw(
        video_frames,
        ball_tracks
    )

    output_video_frames = players_tracks_drawer.draw(
        output_video_frames,
        player_tracks,
        players_assignment,
        ball_acquisition
    )

    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames,
        players_assignment,
        ball_acquisition
    )

    output_video_frames = pass_and_interception_drawer.draw(
        output_video_frames,
        passes,
        interceptions
    )

    output_video_frames = court_keypoints_drawer.draw(
        output_video_frames,
        court_keypoint
    )

    # Court keypoints are ready for transformation (handled internally by converter)

    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        court_keypoint,
        player_tracks=player_tracks,
        ball_tracks=ball_tracks,
        players_assignment=players_assignment,
        converter=tactical_view_converter
    )

    # save video
    save_video(output_video_frames, "output_videos/output_videos.avi")


if __name__ == "__main__":
    main()
