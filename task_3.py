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
        v_max=0.0305,
        v_add=0.0005,
        v_sub=0.0005,
        v_min=0.03,
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

    target_name = "Bill#1"
    photo_num = 0
    # while True:
    #     # 巡航搜索        
    #     search_points = [[-5, 0], [0, 0], [0, 7], [5, 7], [5, 2], [9, 2], [9, -2], [5, -2], [5, -8], [2, -8], [2, -4], [-5, -4]]
    #     photo_interval = 50
    #     photo_count = photo_interval
    #     find_flag = False
    #     pos_now = np.array(flight_controller.getPosition('controller')[:2])
    #     start_num = find_nearest_point_num(pos_now, search_points)
    #     last_position = [4.65, -7.4]
    #     while True:
    #         for target_point_0 in search_points[start_num:]:
    #             target_point = copy.deepcopy(target_point_0)
    #             target_point.append(base_position[2])
    #             target_point = np.array(target_point)
    #             flight_controller.moveTo(target_point, 1, 1, True)
    #             while flight_controller.left_time > 0:
    #                 if photo_count >= photo_interval:
    #                     flight_controller.to_take_photos()
    #                     photo_count = 0
    #                 else:
    #                     photo_count += 1
    #                 result = flight_controller.step_forward_move()
    #                 if 'photos' in result:
    #                     cv2.imwrite("task_3/"+str(photo_num)+"zed0.jpg", result['photos'][0])
    #                     cv2.imwrite("task_3/"+str(photo_num)+"zed1.jpg", result['photos'][1])
    #                     photo_num += 1
    #         start_num = 0

    _, target_handle = vrep.simxGetObjectHandle(clientID, target_name, vrep.simx_opmode_blocking)
    _, target_position = vrep.simxGetObjectPosition(clientID, target_handle, -1, vrep.simx_opmode_blocking)
    target_position[2] = base_position[2]
    target_position = np.array(target_position)
    time_interval = 5
    people_chooser = PeopleChoose(pos_threshold=1, ori_threshold=0, color_threshold=None)
    people_chooser.last_position = target_position[:2]
    print("Target:", target_position)
    while True:
        print(photo_num)
        move_to_position = copy.deepcopy(target_position)
        move_to_position[1] += 3
        flight_controller.setPosition('controller', np.array(move_to_position))
        # flight_controller.moveTo(np.array(move_to_position), 2, 1, True)
        flight_controller.to_take_photos()
        for i in range(time_interval):
            result = flight_controller.step_forward_move()
            if 'photos' in result:
                cv2.imwrite("task_3_2/"+str(photo_num)+"zed0.jpg", result['photos'][0])
                cv2.imwrite("task_3_2/"+str(photo_num)+"zed1.jpg", result['photos'][1])
                photo_0 = cv2.imread("task_3_2/"+str(photo_num)+"zed0.jpg")
                photo_1 = cv2.imread("task_3_2/"+str(photo_num)+"zed1.jpg")
                pos_list = get_people_pos(clientID, photo_1, photo_0)
                photo_num += 1
                next_position = people_chooser.find_next_position(pos_list)
                next_position.append(target_position[2])
                target_position = np.array(next_position)
                print("Target:", target_position)

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
