from utils import read_stub, save_stub
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.append("../")


class TeamAssigner():
    def __init__(self,
                 team_1_class_name="white shirt",
                 team_2_class_name="dark red shirt"
                 ):

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.player_team_dict = {}

    def load_model(self):
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained(
            "patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(
            text=classes, images=pil_image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self, frame, player_bbox, player_id):

        # checking if the player already exists in the dict. Not running the model again
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def get_player_team_across_frame(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):

        self.load_model()

        # read from stub
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        player_assignment = []

        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            if frame_num % 30 == 0:
                self.player_team_dict = {}  # this line fix:
                # handle when two or more players overlap, causing the model to miss-classification
                # => make the model predicts again

            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id)
                player_assignment[frame_num][player_id] = team
        """
        
            The result will be like this:
            player_assignment = [
                {12: 2, 15: 2, 4: 1, 6: 1},
                {12: 2, 15: 2, 4: 1, 6: 1, 8: 2}
            ]
        where the key is player_id and the value is player's team

        """
        save_stub(stub_path, player_assignment)
        return player_assignment
