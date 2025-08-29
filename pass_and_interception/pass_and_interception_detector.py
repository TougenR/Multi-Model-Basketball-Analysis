class PassAndInterceptionDetector():
    def __init__(self):
        pass

    def detect_passes(self, ball_acquisition, player_assignment):
        """
        this module count if there is any pass between two teammate
        Args:
            ball_acquisition (list): a list indicating which player has the ball on each frame
                example:
                ball_acquisition = [-1 , 28, 28, -1]
                    -1 : no player has the ball, 28: player with track_id = 28 has the ball
        
            player_assignment (list): a list of dictionaries indicating which team player is in
                example:
                player_assignment = [
                    {1: 2, 2: 2, 16: 1}
                ]
        Return:
            passes (list): a list indicating if there is a pass occur in this frame
                example:
                    passes = [-1, 1, 2]
                        -1: no pass, 1: team 1 pass, 2: team 2 pass
        """
        passes = [-1] * len(ball_acquisition)
        prev_holder = -1
        prev_frame = -1
        
        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:   # if there is a player has the ball
                prev_holder = ball_acquisition[frame - 1]   
                prev_frame = frame - 1
            
            current_holder = ball_acquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[prev_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes


    def detect_interception(self, ball_acquisition, player_assignment):
        """
        this module count if there is any interception between two different team
        Args:
            ball_acquisition (list): a list indicating which player has the ball on each frame
                example:
                ball_acquisition = [-1 , 28, 28, -1]
                    -1 : no player has the ball, 28: player with track_id = 28 has the ball
        
            player_assignment (list): a list of dictionaries indicating which team player is in
                example:
                player_assignment = [
                    {1: 2, 2: 2, 16: 1}
                ]
        Return:
            interceptions (list): a list indicating if there is a pass occur in this frame
                example:
                    interceptions = [-1, 1, 2]
                        -1: no interception, 1: team 1 interception, 2: team 2 interception
        """
        interceptions = [-1] * len(ball_acquisition)
        prev_holder = -1
        prev_frame = -1
        
        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:   # if there is a player has the ball
                prev_holder = ball_acquisition[frame - 1]   
                prev_frame = frame - 1
            
            current_holder = ball_acquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[prev_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)

                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions       
