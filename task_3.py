from apis_2 import Controller
import vrep
import cv2
import numpy as np
from runnable.distance_metric import zedDistance
from speed_predict import SpeedPredictor
import copy


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
        v_max=0.05,
        v_add=0.0005,
        v_sub=0.0005,
        v_min=0.01,
        v_constant=0.02,
        use_constant_v=False,
        )

    # set required params
    flight_controller.setRequiredParams()

    # set controller position
    base_position = flight_controller.getPosition('base')
    flight_controller.setPosition('controller', base_position)

    # start simulation
    vrep.simxSynchronous(clientID, synchronous_flag)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxSetIntegerSignal(clientID, 'stop', 1, vrep.simx_opmode_oneshot)
    vrep.simxSynchronousTrigger(clientID)

    photo_num = 0
    while True:
        # 巡航搜索        
        search_points = [[-8, -2], [0, -2], [0, 6], [4, 6], [4, -2], [12, -2], [12, -6], [4, -6], [4, -10], [-4, -10], [-8, -10]]
        # search_points = [[-8, -2], [-4, -2], [0, -2], [0, 2], [0, 6], [4, 6], [4, 2], [4, -2], [8, -2], [12, -2], [12, -6], [8, -6], [4, -6], [4, -10], [0, -10], [-4, -10], [-8, -10], [-8, -6]]
        photo_interval = 20
        photo_count = photo_interval
        find_flag = False
        pos_now = np.array(flight_controller.getPosition('controller')[:2])
        start_num = find_nearest_point_num(pos_now, search_points)
        while True:
            for target_point_0 in search_points[start_num:]:
                target_point = copy.deepcopy(target_point_0)
                target_point.append(base_position[2])
                target_point = np.array(target_point)
                flight_controller.moveTo(target_point, 1, 1, True)
                while np.linalg.norm(flight_controller.controller_position - target_point) >= 0.01:
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
