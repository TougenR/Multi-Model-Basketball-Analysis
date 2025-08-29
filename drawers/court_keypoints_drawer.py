import supervision as sv

class CourtKeypointsDrawer():
    def __init__(self):
        self.keypoint_color = "00FF88"  # Bright green color
        self.outline_color = "FFFFFF"   # White outline
    
    def draw(self, video_frame, court_keypoints_detection):        
        # use to draw keypoint outlines (larger radius)
        vertex_outline_annotator = sv.VertexAnnotator(
            color = sv.Color.from_hex(self.outline_color),
            radius=12
        )
        
        # use to draw keypoints (main color, smaller radius)
        vertex_annotator = sv.VertexAnnotator(
            color = sv.Color.from_hex(self.keypoint_color),
            radius=8
        )
        
        # use to label each keypoint with enhanced styling
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color = sv.Color.from_hex("000000"),  # Black background
            text_color = sv.Color.WHITE,
            text_scale=0.8,
            text_thickness=2,
            text_padding=8,
            border_radius=20
        )
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frame):
            annotate_frame = frame.copy()
            keypoints = court_keypoints_detection[frame_num]
            keypoints_np = keypoints.cpu().numpy()
            
            # Draw outline first (larger, white)
            annotate_frame = vertex_outline_annotator.annotate(scene=annotate_frame, key_points=keypoints_np)
            # Draw main keypoints on top (smaller, colored)
            annotate_frame = vertex_annotator.annotate(scene=annotate_frame, key_points=keypoints_np)
            # Add labels with enhanced styling
            annotate_frame = vertex_label_annotator.annotate(scene=annotate_frame, key_points=keypoints_np)        
            output_video_frames.append(annotate_frame)
    
        return output_video_frames
