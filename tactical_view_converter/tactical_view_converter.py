import cv2
from copy import deepcopy
import sys
sys.path.append("../")
from utils import measure_distance
 

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


    def validate_keypoints(self, keypoints_list, distance_threshold=0.8, min_references=2):
        """
        Simple and effective keypoint validation
        Focus on detecting obvious outliers like misplaced middle line
        
        Args:
            keypoints_list: List of detected keypoints for each frame
            distance_threshold: Maximum allowed distance error ratio (default: 80%)
            min_references: Minimum number of reference pairs needed (default: 2)
        """
        keypoints_list = deepcopy(keypoints_list)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]
            
            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_keypoints) 
                              if kp[0] > 0 and kp[1] > 0]
            
            # Need at least 4 detected keypoints for reliable validation
            if len(detected_indices) < 4:
                continue
                
            invalid_keypoints = set()
            
            # Check each keypoint against all others
            for i in detected_indices:
                if i in invalid_keypoints:
                    continue
                    
                failed_checks = 0
                total_checks = 0
                
                # Compare with all other detected keypoints
                for j in detected_indices:
                    if i == j or j in invalid_keypoints:
                        continue
                        
                    # Calculate distance ratio
                    detected_distance = measure_distance(frame_keypoints[i], frame_keypoints[j])
                    template_distance = measure_distance(self.key_points[i], self.key_points[j])
                    
                    if template_distance < 1e-6:  # Skip if template distance too small
                        continue
                        
                    distance_ratio = detected_distance / template_distance
                    
                    # Check for reasonable scale consistency
                    # All distance ratios should be roughly similar (same scale factor)
                    reference_ratios = []
                    
                    # Get scale reference from other keypoint pairs
                    for ref1 in detected_indices:
                        if ref1 == i or ref1 == j or ref1 in invalid_keypoints:
                            continue
                        for ref2 in detected_indices:
                            if ref2 == i or ref2 == j or ref2 == ref1 or ref2 in invalid_keypoints:
                                continue
                                
                            ref_detected = measure_distance(frame_keypoints[ref1], frame_keypoints[ref2])
                            ref_template = measure_distance(self.key_points[ref1], self.key_points[ref2])
                            
                            if ref_template > 1e-6:
                                reference_ratios.append(ref_detected / ref_template)
                            
                            if len(reference_ratios) >= 3:  # Enough references
                                break
                        if len(reference_ratios) >= 3:
                            break
                    
                    if len(reference_ratios) >= min_references:
                        # Calculate median scale factor from references
                        reference_ratios.sort()
                        median_ratio = reference_ratios[len(reference_ratios) // 2]
                        
                        # Check if current distance ratio is consistent with median
                        if median_ratio > 1e-6:
                            error = abs(distance_ratio - median_ratio) / median_ratio
                            
                            if error > distance_threshold:
                                failed_checks += 1
                                
                    total_checks += 1
                
                # Mark keypoint as invalid if it fails too many checks
                if total_checks > 0 and failed_checks / total_checks > 0.6:  # 60% failure rate
                    invalid_keypoints.add(i)
            
            # Special check for middle line (keypoint 7)
            if 7 in detected_indices and 7 not in invalid_keypoints:
                middle_line_pos = frame_keypoints[7]
                
                # Middle line should be roughly in the center horizontally
                frame_center_x = sum(kp[0] for idx, kp in enumerate(frame_keypoints) 
                                   if idx in detected_indices and idx != 7) / max(len(detected_indices) - 1, 1)
                
                # Calculate expected vs actual horizontal position
                horizontal_deviation = abs(middle_line_pos[0] - frame_center_x)
                
                # Estimate court width from detected keypoints
                x_coords = [frame_keypoints[idx][0] for idx in detected_indices if idx != 7]
                if len(x_coords) >= 2:
                    estimated_width = max(x_coords) - min(x_coords)
                    
                    # Middle line should be within 25% of center
                    if horizontal_deviation > estimated_width * 0.25:
                        invalid_keypoints.add(7)
            
            # Apply validation results - remove invalid keypoints
            for i in invalid_keypoints:
                keypoints_list[frame_idx].xy[0][i] *= 0
                keypoints_list[frame_idx].xyn[0][i] *= 0
                
            # Debug output
            if hasattr(self, 'debug') and self.debug and invalid_keypoints:
                print(f"Frame {frame_idx}: Removed invalid keypoints: {list(invalid_keypoints)}")

        return keypoints_list

    def _estimate_scale_factor(self, frame_keypoints, detected_indices):
        """
        Estimate the scale factor between detected frame and template
        using the most reliable keypoint pairs
        """
        scale_ratios = []
        
        # Use corner and edge keypoints as they are most reliable
        reliable_keypoints = [idx for idx in detected_indices 
                             if idx in [0, 5, 6, 7, 10, 15, 16, 17]]  # corners + middle + free throw
        
        if len(reliable_keypoints) < 2:
            reliable_keypoints = detected_indices
        
        for i in range(len(reliable_keypoints)):
            for j in range(i + 1, len(reliable_keypoints)):
                idx1, idx2 = reliable_keypoints[i], reliable_keypoints[j]
                
                detected_dist = measure_distance(frame_keypoints[idx1], frame_keypoints[idx2])
                template_dist = measure_distance(self.key_points[idx1], self.key_points[idx2])
                
                if template_dist > 1e-6:
                    scale_ratios.append(detected_dist / template_dist)
        
        if not scale_ratios:
            return 1.0
        
        # Return median scale ratio
        scale_ratios.sort()
        return scale_ratios[len(scale_ratios) // 2]
