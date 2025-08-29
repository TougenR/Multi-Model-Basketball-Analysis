import cv2
import numpy as np

class PassInterceptionTableDrawer:
    """
    A class responsible for calculating and drawing pass and interception statistics
    in a professional table format on video frames.
    """
    
    def __init__(self, table_position='bottom_center', transparency=0.8):
        """
        Initialize the PassInterceptionTableDrawer.
        
        Args:
            table_position (str): Position of the table ('bottom_center', 'bottom_left', 'bottom_right', 'top_center', etc.)
            transparency (float): Transparency level of the overlay (0.0 to 1.0)
        """
        self.table_position = table_position
        self.transparency = transparency
        self.colors = {
            'background': (255, 255, 255),      # White background
            'border': (50, 50, 50),             # Dark gray border
            'header_bg': (100, 100, 100),       # Gray header background
            'text': (0, 0, 0),                  # Black text
            'header_text': (255, 255, 255),     # White header text
            'team1_color': (0, 100, 200),       # Blue for team 1
            'team2_color': (200, 100, 0),       # Orange for team 2
            'total_color': (50, 150, 50)        # Green for totals
        }

    def get_stats(self, passes, interceptions):
        """
        Calculate the number of passes and interceptions for Team 1 and Team 2.

        Args:
            passes (list): A list of integers representing pass events at each frame.
                (1 represents a pass by Team 1, 2 represents a pass by Team 2, 0 represents no pass.)
            interceptions (list): A list of integers representing interception events at each frame.
                (1 represents an interception by Team 1, 2 represents an interception by Team 2, 0 represents no interception.)

        Returns:
            tuple: A tuple of four integers (team1_pass_total, team2_pass_total,
                team1_interception_total, team2_interception_total) indicating the total
                number of passes and interceptions for both teams.
        """
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(zip(passes, interceptions)):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)
                
            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)
                
        return len(team1_passes), len(team2_passes), len(team1_interceptions), len(team2_interceptions)

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
        
        if self.table_position == 'bottom_center':
            x = (frame_width - table_width) // 2
            y = frame_height - table_height - margin
        elif self.table_position == 'bottom_left':
            x = margin
            y = frame_height - table_height - margin
        elif self.table_position == 'bottom_right':
            x = frame_width - table_width - margin
            y = frame_height - table_height - margin
        elif self.table_position == 'top_center':
            x = (frame_width - table_width) // 2
            y = margin
        elif self.table_position == 'top_left':
            x = margin
            y = margin
        elif self.table_position == 'top_right':
            x = frame_width - table_width - margin
            y = margin
        else:
            # Default to bottom_center
            x = (frame_width - table_width) // 2
            y = frame_height - table_height - margin
            
        return int(x), int(y)

    def draw_table(self, frame, x, y, team1_passes, team2_passes, team1_interceptions, team2_interceptions):
        """
        Draw a professional-looking table with pass and interception statistics.
        
        Args:
            frame (numpy.ndarray): The video frame
            x (int): X coordinate of table top-left corner
            y (int): Y coordinate of table top-left corner
            team1_passes (int): Number of passes by Team 1
            team2_passes (int): Number of passes by Team 2
            team1_interceptions (int): Number of interceptions by Team 1
            team2_interceptions (int): Number of interceptions by Team 2
        """
        # Table dimensions
        table_width = 400
        table_height = 140
        row_height = 35
        
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
        for i in range(1, 5):  # 4 horizontal lines (header + 3 data rows)
            cv2.line(overlay, (x, y + i * row_height), 
                    (x + table_width, y + i * row_height), 
                    self.colors['border'], 1)
        
        # Draw vertical lines
        col_widths = [100, 100, 100, 100]  # Team, Passes, Interceptions, Total
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
        font_scale = 0.6
        header_font_scale = 0.65
        thickness = 1
        header_thickness = 2
        
        # Draw header text
        header_y = y + 25
        cv2.putText(frame, "Team", (x + 25, header_y), font, header_font_scale, 
                   self.colors['header_text'], header_thickness)
        cv2.putText(frame, "Passes", (x + 120, header_y), font, header_font_scale, 
                   self.colors['header_text'], header_thickness)
        cv2.putText(frame, "Intercepts", (x + 210, header_y), font, header_font_scale, 
                   self.colors['header_text'], header_thickness)
        cv2.putText(frame, "Total", (x + 330, header_y), font, header_font_scale, 
                   self.colors['header_text'], header_thickness)
        
        # Calculate totals
        team1_total = team1_passes + team1_interceptions
        team2_total = team2_passes + team2_interceptions
        total_passes = team1_passes + team2_passes
        total_interceptions = team1_interceptions + team2_interceptions
        grand_total = total_passes + total_interceptions
        
        # Draw Team 1 data
        team1_y = y + 60
        cv2.putText(frame, "Team 1", (x + 15, team1_y), font, font_scale, 
                   self.colors['team1_color'], thickness + 1)
        cv2.putText(frame, str(team1_passes), (x + 140, team1_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, str(team1_interceptions), (x + 245, team1_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, str(team1_total), (x + 345, team1_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)
        
        # Draw Team 2 data
        team2_y = y + 95
        cv2.putText(frame, "Team 2", (x + 15, team2_y), font, font_scale, 
                   self.colors['team2_color'], thickness + 1)
        cv2.putText(frame, str(team2_passes), (x + 140, team2_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, str(team2_interceptions), (x + 245, team2_y), font, font_scale, 
                   self.colors['text'], thickness)
        cv2.putText(frame, str(team2_total), (x + 345, team2_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)
        
        # Draw totals row
        total_y = y + 130
        cv2.putText(frame, "Total", (x + 15, total_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)
        cv2.putText(frame, str(total_passes), (x + 140, total_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)
        cv2.putText(frame, str(total_interceptions), (x + 245, total_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)
        cv2.putText(frame, str(grand_total), (x + 345, total_y), font, font_scale, 
                   self.colors['total_color'], thickness + 1)

    def draw_frame(self, frame, frame_num, passes, interceptions):
        """
        Draw a table showing pass and interception statistics on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame
            frame_num (int): The index of the current frame
            passes (list): A list of pass events up to this frame
            interceptions (list): A list of interception events up to this frame

        Returns:
            numpy.ndarray: The frame with the statistics table
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Get stats until current frame
        passes_till_frame = passes[:frame_num + 1]
        interceptions_till_frame = interceptions[:frame_num + 1]
        
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(
            passes_till_frame, 
            interceptions_till_frame
        )
        
        # Calculate table position
        table_width, table_height = 400, 140
        table_x, table_y = self.calculate_table_position(frame_width, frame_height, 
                                                        table_width, table_height)
        
        # Draw the table
        self.draw_table(frame, table_x, table_y, team1_passes, team2_passes, 
                       team1_interceptions, team2_interceptions)
        
        return frame

    def draw(self, video_frames, passes, interceptions):
        """
        Draw pass and interception statistics table on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays) on which to draw
            passes (list): A list of integers representing pass events at each frame
            interceptions (list): A list of integers representing interception events at each frame

        Returns:
            list: A list of frames with pass and interception table drawn on them
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num == 0:
                output_video_frames.append(frame)
                continue
            
            frame_drawn = self.draw_frame(frame.copy(), frame_num, passes, interceptions)
            output_video_frames.append(frame_drawn)
        return output_video_frames

    def get_summary_stats(self, passes, interceptions):
        """
        Get comprehensive summary statistics.
        
        Args:
            passes (list): List of pass events
            interceptions (list): List of interception events
            
        Returns:
            dict: Dictionary containing detailed statistics
        """
        team1_passes, team2_passes, team1_interceptions, team2_interceptions = self.get_stats(passes, interceptions)
        
        total_passes = team1_passes + team2_passes
        total_interceptions = team1_interceptions + team2_interceptions
        
        # Calculate pass accuracy (passes vs interceptions)
        team1_accuracy = (team1_passes / (team1_passes + team2_interceptions)) * 100 if (team1_passes + team2_interceptions) > 0 else 0
        team2_accuracy = (team2_passes / (team2_passes + team1_interceptions)) * 100 if (team2_passes + team1_interceptions) > 0 else 0
        
        return {
            'team1_passes': team1_passes,
            'team2_passes': team2_passes,
            'team1_interceptions': team1_interceptions,
            'team2_interceptions': team2_interceptions,
            'total_passes': total_passes,
            'total_interceptions': total_interceptions,
            'team1_pass_accuracy': team1_accuracy,
            'team2_pass_accuracy': team2_accuracy,
            'pass_to_interception_ratio': total_passes / total_interceptions if total_interceptions > 0 else float('inf')
        }
