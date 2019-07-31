import numpy as np


class PeopleChoose:
    def __init__(self, *args, **kwargs):
        self.speed_history = []
        self.time_interval = kwargs['time_interval']
        self.pos_threshold = kwargs['pos_threshold']
        self.color_threshold = kwargs['color_threshold']
        self.ori_threshold = kwargs['ori_threshold']
        self.last_position = None
        self.last_color = None
        self.last_ori = None

    def find_next_position(self, position_list, color_list):
        possible_result_num = []
        orientation_result = []
        for i in range(len(position_list)):
            if self.last_position is not None:
                orientation_now = (np.array(position_list[i]) - np.array(self.last_position))/np.linalg.norm(np.array(position_list[i]) - np.array(self.last_position))
                orientation_result.append(orientation_now)
                pos_distance = np.linalg.norm(np.array(self.last_position) - np.array(position_list[i]))
                if pos_distance > self.pos_threshold:
                    print(i, "position", pos_distance)
                    continue
            if self.last_color is not None:
                color_distance = np.linalg.norm(np.array(self.last_color)-np.array(color_list[i]))
                if color_distance > self.color_threshold:
                    print(i, "color", color_distance)
                    continue
            possible_result_num.append(i)
        print("Possible result:", possible_result_num)
        if len(possible_result_num) == 0:
            return None
        elif len(possible_result_num) == 1:
            print("Get position", possible_result_num[0], position_list[possible_result_num[0]])
            self.last_position = position_list[possible_result_num[0]]
            self.last_color = color_list[possible_result_num[0]]
            if len(orientation_result) > 0:
                self.last_ori = orientation_result[0]
            return self.last_position
        else:
            min_distance = -1
            min_num = -1
            for i in range(len(possible_result_num)):
                orientation = np.array(position_list[possible_result_num[i]])-np.array(self.last_position)
                orientation = orientation/np.linalg.norm(orientation)
                orientation_distance = orientation.dot(self.last_ori)
                if orientation_distance > min_distance:
                    min_num = i
                    min_distance = orientation_distance
            print("Get position from many", min_num, possible_result_num[min_num], position_list[possible_result_num[min_num]])
            self.last_position = position_list[possible_result_num[min_num]]
            self.last_color = color_list[possible_result_num[min_num]]
            if len(orientation_result) > 0:
                self.last_ori = orientation_result[min_num]
            return self.last_position
