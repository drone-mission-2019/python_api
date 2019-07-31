from apis_task_3 import Controller
import vrep
import cv2
import numpy as np
from runnable.distance_metric import zedDistance
from speed_predict import SpeedPredictor
import copy
from runnable.distance_metric import get_people_pos
from people_choose import PeopleChoose
from speed_predict_task3 import SpeedPredictor


def main():
    # finish first
    vrep.simxFinish(-1)

    # connect to server
    clientID = vrep.simxStart("127.0.0.1", 19997, True, True, 5000, 5)
    if clientID != -1:
        print("Connect Succesfully.")
    else:
        print("Connect failed.")
        assert 0
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # get handles
    _, vision_handle_0 = vrep.simxGetObjectHandle(clientID, "zed_vision0", vrep.simx_opmode_blocking)
    _, vision_handle_1 = vrep.simxGetObjectHandle(clientID, "zed_vision1", vrep.simx_opmode_blocking)
    _, controller_handle= vrep.simxGetObjectHandle(clientID, "Quadricopter_target", vrep.simx_opmode_blocking)
    _, base_handle = vrep.simxGetObjectHandle(clientID, "Quadricopter", vrep.simx_opmode_blocking)

    # set Controller
    synchronous_flag = True
    time_interval = 0.05
    flight_controller = Controller(
        clientID=clientID,
        base_handle=base_handle, 
        controller_handle=controller_handle, 
        vision_handle_0=vision_handle_0, 
        vision_handle_1=vision_handle_1,
        synchronous=synchronous_flag,
        time_interval=time_interval,
        v_max=0.2,
        v_add=0.0005,
        v_sub=0.0005,
        v_min=0.02,
        v_constant=0.02,
        use_constant_v=False,
        )

    # set controller position
    base_position = flight_controller.getPosition('base')
    flight_controller.setPosition('controller', base_position)

    # start simulation
    vrep.simxSynchronous(clientID, synchronous_flag)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(clientID)

    # photo_interval = 5
    # last_position = [4.65, -7.4]
    # target_position = copy.deepcopy(last_position)
    # target_position.append(base_position[2])
    # target_position = np.array(target_position)
    # speed_predictor = SpeedPredictor(4, None, max_predict_step=1, extra_time=0)
    # flight_controller.moveTo(target_position, 2, 1, True)
    # photo_num = 0
    # position_getter = PeopleChoose(time_interval=5, pos_threshold=2, color_threshold=15, ori_threshold=0.5)
    # while True:
    #     flight_controller.to_take_photos()
    #     print("To take photos")
    #     for iiii in range(photo_interval):
    #         result = flight_controller.step_forward_move()
    #         speed_predictor.step_forward()
    #         if 'photos' in result:
    #             print("Photo num", photo_num)
    #             cv2.imwrite("task_3/"+str(photo_num)+"zed0.jpg", result['photos'][0])
    #             cv2.imwrite("task_3/"+str(photo_num)+"zed1.jpg", result['photos'][1])
    #             pos_list, color_list = get_people_pos(clientID, result['photos'][1], result['photos'][0], last_position)
    #             print(pos_list, color_list)
    #             if pos_list is None or color_list is None or len(pos_list) == 0 or len(color_list) == 0:
    #                 continue
    #             pos_new = position_getter.find_next_position(pos_list, color_list)
    #             if pos_new is not None:
    #                 last_position = pos_new
    #             print("People pos", last_position)
    #             if last_position is not None:
    #                 target_position = copy.deepcopy(last_position)
    #                 target_position.append(base_position[2])
    #                 target_position = np.array(target_position)

    #                 speed_predictor.give_new_information(np.array(target_position))
    #                 new_target_position = speed_predictor.get_next_target(flight_controller.controller_position, flight_controller.left_time)
    #                 if new_target_position is not None:
    #                     left_time_before = flight_controller.left_time
    #                     flight_controller.moveTo(new_target_position, 1, 1, True)
    #                     flag = True  # 防止不收敛情况出现
    #                     count = 0
    #                     while flight_controller.left_time != left_time_before :
    #                         if count >= 10:
    #                             flag = False
    #                             break
    #                         new_target_position = speed_predictor.get_next_target(flight_controller.controller_position, flight_controller.left_time)
    #                         if new_target_position is None:
    #                             flag = False
    #                             break
    #                         left_time_before = flight_controller.left_time
    #                         flight_controller.moveTo(new_target_position, 1, 1, True)
    #                         count += 1
    #                     if flag:
    #                         target_position = new_target_position
                    
    #                 flight_controller.moveTo(target_position, 1, 1, True)
    #     photo_num += 1

    target_name = "Bill#1"
    photo_num = 0
    while True:
        # 巡航搜索        
        search_points = [[-5, 0], [0, 0], [0, 7], [5, 7], [5, 2], [9, 2], [9, -2], [5, -2], [5, -8], [2, -8], [2, -4], [0, -4], [-5, -4]]
        photo_interval = 50
        photo_count = photo_interval
        find_flag = False
        pos_now = np.array(flight_controller.getPosition('controller')[:2])
        start_num = find_nearest_point_num(pos_now, search_points)
        last_position = [4.65, -7.4]
        while True:
            for target_point_0 in search_points[start_num:]:
                target_point = copy.deepcopy(target_point_0)
                target_point.append(base_position[2])
                target_point = np.array(target_point)
                flight_controller.moveTo(target_point, 1, 1, True)
                while flight_controller.left_time > 0:
                    if photo_count >= photo_interval:
                        flight_controller.to_take_photos()
                        photo_count = 0
                    else:
                        photo_count += 1
                    result = flight_controller.step_forward_move()
                    if 'photos' in result:
                        cv2.imwrite("task_3/"+str(photo_num)+"zed0.jpg", result['photos'][0])
                        cv2.imwrite("task_3/"+str(photo_num)+"zed1.jpg", result['photos'][1])
                        photo_num += 1
            start_num = 0


def find_nearest_point_num(pos_now, search_points):
    distance_min = -1
    num_min = -1
    for num in range(len(search_points)):
        delta_distance = np.linalg.norm(search_points[num] - pos_now)
        if delta_distance < distance_min or distance_min == -1:
            distance_min = delta_distance
            num_min = num
    return num_min


if __name__ == '__main__':
    main()
