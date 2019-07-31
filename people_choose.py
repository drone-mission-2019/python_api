import numpy as np


class PeopleChoose:
    def __init__(self, *args, **kwargs):
        self.speed_history = []
        self.pos_threshold = kwargs['pos_threshold']
        self.color_threshold = kwargs['color_threshold']
        self.ori_threshold = kwargs['ori_threshold']
        self.last_position = None
        self.last_color = None
        self.last_ori = None

    def find_next_position(self, position_list):
        min_distance = -1
        min_pos = None
        for _pos in position_list:
            pos = _pos[:2]
            pos_distance = np.linalg.norm(np.array(pos) - np.array(self.last_position))
            if min_pos is None or pos_distance < min_distance:
                min_pos = pos
                min_distance = pos_distance
        self.last_position = min_pos.copy()
        return min_pos
