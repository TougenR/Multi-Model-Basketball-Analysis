import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tactical_view_converter import TacticalViewConverterOptimized


class TacticalViewDrawer:
    """
    Enhanced tactical view drawer with perspective transformation capabilities.

    Supports both traditional overlay mode and perspective-transformed tactical view
    with player and ball position mapping.
    """

    def __init__(self,
                 perspective_mode: bool = False,
                 overlay_position: Tuple[int, int] = (450, 20),
                 transparency: float = 0.6):
        """
        Initialize tactical view drawer.

        Args:
            perspective_mode: If True, use perspective transformation instead of overlay
            overlay_position: Position for overlay mode (x, y)
            transparency: Transparency level for overlay (0.0 to 1.0)
        """
        self.perspective_mode = perspective_mode
        self.start_x, self.start_y = overlay_position
        self.transparency = transparency

        # Colors for visualization
        self.colors = {
            'team1': (255, 0, 0),      # Red
            'team2': (0, 0, 255),      # Blue
            'ball': (255, 255, 0),     # Yellow
            'keypoint': (0, 255, 0),   # Green
            'text': (255, 255, 255)    # White
        }

    def draw(self,
             video_frames: List[np.ndarray],
             court_image_path: str,
             width: int,
             height: int,
             court_keypoints_list: List,
             player_tracks: Optional[List] = None,
             ball_tracks: Optional[List] = None,
             players_assignment: Optional[List] = None,
             converter: Optional[TacticalViewConverterOptimized] = None) -> List[np.ndarray]:
        """
        Draw tactical view on video frames.

        Args:
            video_frames: List of input video frames
            court_image_path: Path to court image
            width: Width of tactical view
            height: Height of tactical view
            court_keypoints_list: List of detected keypoints for each frame
            player_tracks: Optional player tracking data
            ball_tracks: Optional ball tracking data
            players_assignment: Optional team assignment data
            converter: Optional tactical view converter for perspective mode

        Returns:
            List of video frames with tactical view overlay
        """
        if self.perspective_mode and converter is not None:
            try:
                return self._draw_perspective_view(
                    video_frames, court_image_path, width, height,
                    court_keypoints_list, player_tracks, ball_tracks, players_assignment, converter
                )
            except Exception as e:
                print(f"Warning: Perspective view failed, falling back to overlay mode: {e}")
                # Fall back to overlay mode if perspective fails
                return self._draw_overlay_view(
                    video_frames, court_image_path, width, height, court_keypoints_list
                )
        else:
            return self._draw_overlay_view(
                video_frames, court_image_path, width, height, court_keypoints_list
            )

    def _draw_overlay_view(self,
                          video_frames: List[np.ndarray],
                          court_image_path: str,
                          width: int,
                          height: int,
                          court_keypoints_list: List) -> List[np.ndarray]:
        """
        Draw traditional overlay tactical view.

        Args:
            video_frames: List of input video frames
            court_image_path: Path to court image
            width: Width of tactical view
            height: Height of tactical view
            court_keypoints_list: List of detected keypoints for each frame

        Returns:
            List of video frames with overlay tactical view
        """
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            # Define overlay region
            x1 = self.start_x
            y1 = self.start_y
            x2 = x1 + width
            y2 = y1 + height

            # Ensure overlay region is within frame bounds
            frame_h, frame_w = output_frame.shape[:2]
            x2 = min(x2, frame_w)
            y2 = min(y2, frame_h)

            if x1 < x2 and y1 < y2:
                # Apply overlay with transparency
                overlay = output_frame[y1:y2, x1:x2].copy()
                court_resized = court_image[:y2-y1, :x2-x1]

                cv2.addWeighted(
                    court_resized, self.transparency,
                    overlay, 1 - self.transparency, 0,
                    output_frame[y1:y2, x1:x2]
                )

            # Draw keypoints for this frame
            if frame_idx < len(court_keypoints_list):
                frame_keypoints = court_keypoints_list[frame_idx]
                if hasattr(frame_keypoints, 'xy'):
                    keypoints = frame_keypoints.xy[0].tolist()
                else:
                    keypoints = frame_keypoints

                for keypoint_idx, keypoint in enumerate(keypoints):
                    if len(keypoint) >= 2 and keypoint[0] > 0 and keypoint[1] > 0:
                        x, y = keypoint
                        x += self.start_x
                        y += self.start_y
                        center = (int(x), int(y))

                        # Draw keypoint
                        cv2.circle(output_frame, center, radius=5,
                                 color=self.colors['keypoint'], thickness=-1)

                        # Draw keypoint number
                        cv2.putText(output_frame, str(keypoint_idx), center,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.colors['text'], 2)

            output_video_frames.append(output_frame)

        return output_video_frames

    def _draw_perspective_view(self,
                             video_frames: List[np.ndarray],
                             court_image_path: str,
                             width: int,
                             height: int,
                             court_keypoints_list: List,
                             player_tracks: Optional[List],
                             ball_tracks: Optional[List],
                             players_assignment: Optional[List],
                             converter: TacticalViewConverterOptimized) -> List[np.ndarray]:
        """
        Draw perspective-transformed tactical view with player and ball positions.

        Args:
            video_frames: List of input video frames
            court_image_path: Path to court image
            width: Width of tactical view
            height: Height of tactical view
            court_keypoints_list: List of detected keypoints for each frame
            player_tracks: Player tracking data
            ball_tracks: Ball tracking data
            converter: Tactical view converter for perspective transformation

        Returns:
            List of video frames with perspective tactical view
        """
        # Load and prepare court background
        court_image = cv2.imread(court_image_path)
        court_image = cv2.resize(court_image, (width, height))

        output_video_frames = []
        for frame_idx, frame in enumerate(video_frames):
            output_frame = frame.copy()

            # Position tactical view at the top center of the frame
            frame_h, frame_w = output_frame.shape[:2]
            tactical_x = (frame_w - width) // 2  # Center horizontally
            tactical_y = 20  # Position at top

            # Copy court background with transparency
            x1, y1 = tactical_x, tactical_y
            x2, y2 = x1 + width, y1 + height

            # Ensure tactical view fits within frame bounds
            if x2 > frame_w:
                # Scale down if too wide
                scale = (frame_w - 40) / width
                width = int(width * scale)
                height = int(height * scale)
                court_image = cv2.resize(court_image, (width, height))
                tactical_x = (frame_w - width) // 2
                x1, y1 = tactical_x, tactical_y
                x2, y2 = x1 + width, y1 + height

            if y2 > frame_h:
                # Scale down if too tall
                scale = (frame_h - 60) / height  # Leave space at bottom
                width = int(width * scale)
                height = int(height * scale)
                court_image = cv2.resize(court_image, (width, height))
                tactical_x = (frame_w - width) // 2
                x1, y1 = tactical_x, tactical_y
                x2, y2 = x1 + width, y1 + height

            if x1 >= 0 and y1 >= 0 and x2 <= frame_w and y2 <= frame_h:
                # Apply transparency overlay
                overlay_region = output_frame[y1:y2, x1:x2]
                cv2.addWeighted(
                    court_image, self.transparency,
                    overlay_region, 1 - self.transparency, 0,
                    output_frame[y1:y2, x1:x2]
                )

                # Draw hardcoded court keypoints on tactical view
                # Following the same approach as original tactical_view_converter.py
                template_keypoints = converter.key_points  # Get hardcoded template keypoints

                # Draw all 18 court keypoints with appropriate styling
                for keypoint_idx, keypoint in enumerate(template_keypoints):
                    x, y = keypoint
                    x += tactical_x  # Add tactical view offset for overlay positioning
                    y += tactical_y
                    center = (int(x), int(y))

                    # Different colors for different court areas
                    if keypoint_idx in [0, 5, 10, 15]:  # Corner points
                        color = (0, 255, 255)  # Yellow for corners
                        radius = 4
                    elif keypoint_idx in [6, 7]:  # Middle line points
                        color = (255, 0, 255)  # Magenta for center line
                        radius = 4
                    elif keypoint_idx in [8, 9, 16, 17]:  # Free throw line points
                        color = (255, 165, 0)  # Orange for free throw
                        radius = 3
                    elif keypoint_idx in [1, 2, 3, 4, 11, 12, 13, 14]:  # Edge points
                        color = (0, 255, 0)    # Green for edges
                        radius = 2
                    else:
                        color = self.colors['keypoint']  # Default green
                        radius = 2

                    # Draw the keypoint
                    cv2.circle(output_frame, center, radius=radius,
                             color=color, thickness=-1)

                    # Draw connecting lines for court structure
                    if keypoint_idx == 0:  # Top-left corner, draw to next edge point
                        next_point = template_keypoints[1]
                        next_x, next_y = next_point
                        next_x += tactical_x
                        next_y += tactical_y
                        cv2.line(output_frame, center, (int(next_x), int(next_y)),
                                color, thickness=1)

                    elif keypoint_idx == 5:  # Bottom-left corner, draw to bottom center
                        next_point = template_keypoints[6]
                        next_x, next_y = next_point
                        next_x += tactical_x
                        next_y += tactical_y
                        cv2.line(output_frame, center, (int(next_x), int(next_y)),
                                color, thickness=1)

                    elif keypoint_idx == 6:  # Bottom center, draw to bottom-right corner
                        next_point = template_keypoints[10]
                        next_x, next_y = next_point
                        next_x += tactical_x
                        next_y += tactical_y
                        cv2.line(output_frame, center, (int(next_x), int(next_y)),
                                color, thickness=1)

                    elif keypoint_idx == 10:  # Bottom-right corner, draw to next edge point
                        next_point = template_keypoints[11]
                        next_x, next_y = next_point
                        next_x += tactical_x
                        next_y += tactical_y
                        cv2.line(output_frame, center, (int(next_x), int(next_y)),
                                color, thickness=1)

                    elif keypoint_idx == 15:  # Top-right corner, draw to next edge point
                        next_point = template_keypoints[14]
                        next_x, next_y = next_point
                        next_x += tactical_x
                        next_y += tactical_y
                        cv2.line(output_frame, center, (int(next_x), int(next_y)),
                                color, thickness=1)

                    # Draw keypoint labels for important points
                    if keypoint_idx in [6, 7]:  # Only label center line points
                        cv2.putText(output_frame, f"C{keypoint_idx}", center,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                  (255, 255, 255), 1)

                # Draw additional court boundary lines for better visualization
                # Draw left and right boundaries
                left_top = (template_keypoints[0][0] + tactical_x, template_keypoints[0][1] + tactical_y)
                left_bottom = (template_keypoints[5][0] + tactical_x, template_keypoints[5][1] + tactical_y)
                right_top = (template_keypoints[15][0] + tactical_x, template_keypoints[15][1] + tactical_y)
                right_bottom = (template_keypoints[10][0] + tactical_x, template_keypoints[10][1] + tactical_y)

                cv2.line(output_frame, left_top, left_bottom, (255, 255, 255), thickness=2)  # Left boundary
                cv2.line(output_frame, right_top, right_bottom, (255, 255, 255), thickness=2)  # Right boundary

                # Draw top and bottom boundaries
                cv2.line(output_frame, left_top, right_top, (255, 255, 255), thickness=2)  # Top boundary
                cv2.line(output_frame, left_bottom, right_bottom, (255, 255, 255), thickness=2)  # Bottom boundary

                # Draw center line
                center_top = (template_keypoints[7][0] + tactical_x, template_keypoints[7][1] + tactical_y)
                center_bottom = (template_keypoints[6][0] + tactical_x, template_keypoints[6][1] + tactical_y)
                cv2.line(output_frame, center_top, center_bottom, (255, 0, 255), thickness=2)  # Center line

                # Draw free throw lines
                left_ft_top = (template_keypoints[8][0] + tactical_x, template_keypoints[8][1] + tactical_y)
                left_ft_bottom = (template_keypoints[9][0] + tactical_x, template_keypoints[9][1] + tactical_y)
                right_ft_top = (template_keypoints[16][0] + tactical_x, template_keypoints[16][1] + tactical_y)
                right_ft_bottom = (template_keypoints[17][0] + tactical_x, template_keypoints[17][1] + tactical_y)

                cv2.line(output_frame, left_ft_top, left_ft_bottom, (255, 165, 0), thickness=1)  # Left free throw
                cv2.line(output_frame, right_ft_top, right_ft_bottom, (255, 165, 0), thickness=1)  # Right free throw

                # Draw players in tactical view
                if player_tracks is not None:
                    # Get detected keypoints for this frame for transformation
                    frame_keypoints = court_keypoints_list[frame_idx]
                    if hasattr(frame_keypoints, 'xy'):
                        detected_keypoints = frame_keypoints.xy[0].cpu().numpy()
                    else:
                        detected_keypoints = np.array(frame_keypoints)

                    tactical_players = converter.transform_players_to_tactical_view(
                        player_tracks, detected_keypoints, frame_idx, (frame_h, frame_w)
                    )

                    if tactical_players:
                        # Get team assignment for this frame
                        frame_team_assignment = players_assignment[frame_idx] if (players_assignment and frame_idx < len(players_assignment)) else {}

                        # Team assignments applied successfully

                        for player_id, player_data in tactical_players.items():
                            pos = player_data['position']

                            # Get team assignment from the team assignment data
                            team = frame_team_assignment.get(player_id, 1)  # Default to team 1 if not found

                            # Choose color based on team
                            if team == 1:
                                color = self.colors['team1']  # Red for team 1
                            elif team == 2:
                                color = self.colors['team2']  # Blue for team 2
                            else:
                                color = (128, 128, 128)  # Gray for unknown team

                            # Convert to tactical view coordinates (pos should already be in tactical coordinates)
                            # No need to add tactical_x here because the transformation should map to tactical view space
                            center = (int(pos[0] + tactical_x), int(pos[1] + tactical_y))

                            # Draw player circle with different styles for different teams
                            cv2.circle(output_frame, center, radius=8,
                                     color=color, thickness=-1)

                            # Add white border for better visibility
                            cv2.circle(output_frame, center, radius=8,
                                     color=(255, 255, 255), thickness=2)

                            # Draw player number with contrasting color
                            text_color = (255, 255, 255) if team == 2 else (0, 0, 0)  # White text on blue, black on red
                            cv2.putText(output_frame, str(player_id % 100), center,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                      text_color, 1)

                            # Player colors applied based on team assignment

                # Draw ball in tactical view
                if ball_tracks is not None:
                    # Get detected keypoints for this frame for transformation
                    frame_keypoints = court_keypoints_list[frame_idx]
                    if hasattr(frame_keypoints, 'xy'):
                        detected_keypoints = frame_keypoints.xy[0].cpu().numpy()
                    else:
                        detected_keypoints = np.array(frame_keypoints)

                    tactical_ball = converter.transform_ball_to_tactical_view(
                        ball_tracks, detected_keypoints, frame_idx, (frame_h, frame_w)
                    )

                    if tactical_ball:
                        pos = tactical_ball['position']
                        center = (int(pos[0] + tactical_x), int(pos[1] + tactical_y))

                        # Ball position debug (removed for cleaner output)

                        # Draw ball
                        cv2.circle(output_frame, center, radius=6,
                                 color=self.colors['ball'], thickness=-1)
                        cv2.circle(output_frame, center, radius=6,
                                 color=(255, 255, 255), thickness=1)

                # Draw border around tactical view
                cv2.rectangle(output_frame, (x1, y1), (x2-1, y2-1), (255, 255, 255), 2)

                # Add label
                cv2.putText(output_frame, "Tactical View", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            output_video_frames.append(output_frame)

        return output_video_frames

    def set_perspective_mode(self, enabled: bool):
        """Enable or disable perspective mode."""
        self.perspective_mode = enabled

    def set_overlay_position(self, x: int, y: int):
        """Set overlay position for traditional mode."""
        self.start_x, self.start_y = x, y

    def set_transparency(self, transparency: float):
        """Set overlay transparency (0.0 to 1.0)."""
        self.transparency = max(0.0, min(1.0, transparency))

        
