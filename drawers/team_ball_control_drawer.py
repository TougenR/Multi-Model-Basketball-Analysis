import cv2 
import numpy as np

class TeamBallControlDrawer:
    """
    A class responsible for calculating and drawing team ball control statistics 
    in a table format on video frames.
    """
    
    def __init__(self, table_position='top_left', transparency=0.8):
        """
        Initialize the TeamBallControlDrawer.
        
        Args:
            table_position (str): Position of the table ('bottom_right', 'bottom_left', 'top_right', 'top_left')
            transparency (float): Transparency level of the overlay (0.0 to 1.0)
        """
        self.table_position = table_position
        self.transparency = transparency
        self.colors = {
            'background': (255, 255, 255),  # White background
            'border': (50, 50, 50),         # Dark gray border
            'header_bg': (100, 100, 100),   # Gray header background
            'text': (0, 0, 0),              # Black text
            'header_text': (255, 255, 255), # White header text
            'team1_color': (0, 100, 200),   # Blue for team 1
            'team2_color': (200, 100, 0)    # Orange for team 2
        }

    def get_team_ball_control(self, player_assignment, ball_acquisition):
        """
        Calculate which team has ball control for each frame.

        Args:
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            numpy.ndarray: An array indicating which team has ball control for each frame
                (1 for Team 1, 2 for Team 2, -1 for no control).
        """
        team_ball_control = []
        for player_assignment_frame, ball_acquisition_frame in zip(player_assignment, ball_acquisition):
            if ball_acquisition_frame == -1:
                team_ball_control.append(-1)
                continue
            if ball_acquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue
            if player_assignment_frame[ball_acquisition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        return np.array(team_ball_control)

    def calculate_table_position(self, frame_width, frame_height, table_width, table_height):
        """
        Calculate the position of the table based on the specified position.
        
        Args:
            frame_width (int): Width of the video frame
            frame_height (int): Height of the video frame  
            table_width (int): Width of the table
            table_height (int): Height of the table
            
        Returns:
            tuple: (x, y) coordinates for the top-left corner of the table
        """
        margin = 20
        
        if self.table_position == 'bottom_right':
            x = frame_width - table_width - margin
            y = frame_height - table_height - margin
        elif self.table_position == 'bottom_left':
            x = margin
            y = frame_height - table_height - margin
        elif self.table_position == 'top_right':
            x = frame_width - table_width - margin
            y = margin
        elif self.table_position == 'top_left':
            x = margin
            y = margin
        else:
            # Default to bottom_right
            x = frame_width - table_width - margin
            y = frame_height - table_height - margin
            
        return int(x), int(y)

    def draw_table(self, frame, x, y, team1_pct, team2_pct, total_frames):
        """
        Draw a professional-looking table with ball control statistics.
        
        Args:
            frame (numpy.ndarray): The video frame
            x (int): X coordinate of table top-left corner
            y (int): Y coordinate of table top-left corner
            team1_pct (float): Team 1 ball control percentage
            team2_pct (float): Team 2 ball control percentage
            total_frames (int): Total number of frames processed
        """
        # Table dimensions
        table_width = 280
        table_height = 120
        row_height = 30
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw main table background
        cv2.rectangle(overlay, (x, y), (x + table_width, y + table_height), 
                     self.colors['background'], -1)
        
        # Draw table border
        cv2.rectangle(overlay, (x, y), (x + table_width, y + table_height), 
                     self.colors['border'], 2)
        
        # Draw header background
        cv2.rectangle(overlay, (x, y), (x + table_width, y + row_height), 
                     self.colors['header_bg'], -1)
        
        # Draw horizontal lines
        for i in range(1, 4):  # 3 horizontal lines (header + 2 data rows)
            cv2.line(overlay, (x, y + i * row_height), 
                    (x + table_width, y + i * row_height), 
                    self.colors['border'], 1)
        
        # Draw vertical lines
        col_widths = [100, 90, 90]  # Team, Control %, Time
        col_x = x
        for width in col_widths:
            cv2.line(overlay, (col_x, y), (col_x, y + table_height), 
                    self.colors['border'], 1)
            col_x += width
        cv2.line(overlay, (col_x, y), (col_x, y + table_height), 
                self.colors['border'], 1)  # Right border
        
        # Apply transparency
        cv2.addWeighted(overlay, self.transparency, frame, 1 - self.transparency, 0, frame)
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Draw header text
        header_y = y + 20
        cv2.putText(frame, "Team", (x + 25, header_y), font, font_scale, 
                   self.colors['header_text'], thickness)
        cv2.putText(frame, "Control %", (x + 125, header_y), font, font_scale, 
                   self.colors['header_text'], thickness)
        cv2.putText(frame, "Time", (x + 215, header_y), font, font_scale, 
                   self.colors['header_text'], thickness)
        
        # Calculate time in possession (assuming 30 fps)
        fps = 30
        team1_time = int((team1_pct / 100.0) * total_frames / fps)
        team2_time = int((team2_pct / 100.0) * total_frames / fps)
        
        # Draw Team 1 data
        team1_y = y + 50
        cv2.putText(frame, "Team 1", (x + 15, team1_y), font, font_scale, 
                   self.colors['team1_color'], thickness + 1)
        cv2.putText(frame, f"{team1_pct:.1f}%", (x + 135, team1_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, f"{team1_time}s", (x + 225, team1_y), font, font_scale, 
                   self.colors['text'], thickness)
        
        # Draw Team 2 data
        team2_y = y + 80
        cv2.putText(frame, "Team 2", (x + 15, team2_y), font, font_scale, 
                   self.colors['team2_color'], thickness + 1)
        cv2.putText(frame, f"{team2_pct:.1f}%", (x + 135, team2_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, f"{team2_time}s", (x + 225, team2_y), font, font_scale, 
                   self.colors['text'], thickness)
        
        # Draw summary info
        summary_y = y + 110
        no_control_pct = 100.0 - team1_pct - team2_pct
        cv2.putText(frame, f"No Control: {no_control_pct:.1f}%", (x + 15, summary_y), 
                   font, 0.4, self.colors['text'], 1)

    def draw_frame(self, frame, frame_num, team_ball_control):
        """
        Draw a table showing team ball control percentages on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame
            frame_num (int): The index of the current frame
            team_ball_control (numpy.ndarray): Array indicating team control for each frame

        Returns:
            numpy.ndarray: The frame with the ball control table
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate statistics up to current frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        total_frames = len(team_ball_control_till_frame)
        
        if total_frames == 0:
            return frame
        
        # Calculate ball control statistics
        team_1_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_frames = np.sum(team_ball_control_till_frame == 2)
        
        team1_pct = (team_1_frames / total_frames) * 100
        team2_pct = (team_2_frames / total_frames) * 100
        
        # Calculate table position
        table_width, table_height = 280, 120
        table_x, table_y = self.calculate_table_position(frame_width, frame_height, 
                                                        table_width, table_height)
        
        # Draw the table
        self.draw_table(frame, table_x, table_y, team1_pct, team2_pct, total_frames)
        
        return frame

    def draw(self, video_frames, player_assignment, ball_acquisition):
        """
        Draw team ball control statistics table on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays) on which to draw
            player_assignment (list): A list of dictionaries indicating team assignments
            ball_acquisition (list): A list indicating which player has possession of the ball

        Returns:
            list: A list of frames with team ball control table drawn on them
        """
        team_ball_control = self.get_team_ball_control(player_assignment, ball_acquisition)
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                output_video_frames.append(frame)
                continue
                
            frame_drawn = self.draw_frame(frame.copy(), frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)
            
        return output_video_frames


