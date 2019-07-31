import cv2
import numpy as np
from scipy.spatial.distance import pdist
from skimage.measure import compare_ssim
import dlib
import sys
import os
import math
from runnable.distance_metric import *

person = []
person_descriptors = []

person.append(cv2.imread('pictures/face_1.png'))
person.append(cv2.imread('pictures/face_2.png'))
person.append(cv2.imread('pictures/face_3.png'))
person.append(cv2.imread('pictures/face_4.png'))
person.append(cv2.imread('pictures/face_5.png'))
person.append(cv2.imread('pictures/face_6.png'))

# person1 = cv2.cvtColor(person1, cv2.COLOR_BGR2GRAY)
# person2 = cv2.cvtColor(person2, cv2.COLOR_BGR2GRAY)
# person3 = cv2.cvtColor(person3, cv2.COLOR_BGR2GRAY)
# person4 = cv2.cvtColor(person4, cv2.COLOR_BGR2GRAY)
# person5 = cv2.cvtColor(person5, cv2.COLOR_BGR2GRAY)
# person6 = cv2.cvtColor(person6, cv2.COLOR_BGR2GRAY)

predictor_path = 'face_recognition/shape_predictor.dat'
face_rec_model_path = 'face_recognition/dlib_face_recognition.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

for i in range(0, 6):
    dets = detector(person[i], 1)
    for k, d in enumerate(dets): 
        shape = sp(person[i], d)
        face_descriptor = facerec.compute_face_descriptor(person[i], shape)
        v = np.array(face_descriptor)
        person_descriptors.append(v)


def resNet(zed1, zed0, personID, clientID):
    """
    给出当前拍摄到的图像，和所给的id进行匹配，检测是否是当前的人脸
    :param zed1: 无人机相机左目视觉
    :param zed0: 无人机相机右目视觉
    :type  ndarrays
    :param personID: 目标人物的id
    :type  int

    :return: 人脸的位置，相当于世界坐标系
    :rtype: [x, y, z] with respect to the world
            return None for no faces in the picture
    """
    zed1 = rotate_image_trivial(zed1, 180)
    zed0 = rotate_image_trivial(zed0, 180)

    h1, w1 = zed1.shape[:2]
    h0, w0 = zed0.shape[:2]
    zed1_expanded = resize_image(zed1, w1 * 2, h1 * 2)
    zed0_expanded = resize_image(zed0, w0 * 2, h0 * 2)
    min_dist = 10000
    min_id = -1
    index = 0
    faces = detector(zed1_expanded, 1)
    faces0 = detector(zed0_expanded, 1)

    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0

    for k, d in enumerate(faces):
        index += 1
        zed_shape = sp(zed1_expanded, d)
        zed_descriptor = facerec.compute_face_descriptor(zed1_expanded, zed_shape)
        zed_v = np.array(zed_descriptor)
        print(len(person_descriptors))
        for i in range(0, len(person_descriptors)):
            dist = np.linalg.norm(person_descriptors[i]-zed_v)
            if dist < min_dist:
                min_dist = dist
                min_id = i
        print(min_id)
        if min_id == personID:
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            x1 = (left/2 + right/2) / 2
            y1 = (top/2 + bottom/2) / 2
            break

    if x1 == 0 and y1 == 0:
        return None

    index0 = 0
    for k, d in enumerate(faces0):
        index0 += 1
        if index0 == index:
            left = d.left()
            top = d.top()
            right = d.right()
            bottom = d.bottom()
            x0 = (left/2 + right/2) / 2
            y0 = (top/2 + bottom/2) / 2
            break
    print(x0, y0, x1, y1)
    return reprojectionTo3D(clientID, [x1, y1], [x0, y0])

def display_image(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def rotate_image_reverse(coordinate, M):
    A = M[:, 0:2] 
    B = M[:, 2]
    coordinate -= B
    A_inv = np.linalg.inv(A)
    return np.dot(A_inv, np.transpose(coordinate))


def rotate_image_trivial(img, angle):
    (h,w) = img.shape[:2]
    center = (w / 2,h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(img, M, (w,h))
    return rotated


def rotate_image(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2) 

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(img, M, (nW, nH))
    return rotated, M

def trivial_rotate_image(img, angle):
    (h,w) = img.shape[:2]
    center = (w / 2,h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(img, M, (w,h))
    return rotated

def resize_image(img, w, h):
    return cv2.resize(img, (w, h))

def mahalanobis_distance(vec0, vec1):
    combo = np.array([vec0, vec1])
    return pdist(combo, 'mahalanobis')

def edge_detector(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    return lap_gray

def face_detector_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lap = cv2.Laplacian(img,cv2.CV_64F)#拉普拉斯边缘检测 
    # lap = np.uint8(np.absolute(lap))##对lap去绝对值
    # lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.01, 2)
    return faces

def face_detector_dlib(img):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    return dets

def face_selector(img):
    checkin = False
    angles = 30
    while True:
        angles -= 30
        processed_img, M = rotate_image(img, angles)
        # faces = face_detector_dlib(processed_img)
        # for index, face in enumerate(faces):
        #     checkin = True
        #     left = face.left()
        #     top = face.top()
        #     right = face.right()
        #     bottom = face.bottom()
        #     cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 3)
        faces = face_detector_opencv(processed_img)
        for (x, y, w, h) in faces:
            checkin = True
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0))
        if checkin == True or angles <= -360:
            break

    if checkin == True:
        return faces, angles, processed_img, M
    else:
        return None, None, None, None

def find_peopleID(clientID, zed1, zed0, id):
    """
    给出当前拍摄到的图像，和所给的id进行匹配，检测是否是当前的人脸
    :param zed1: 无人机相机左目视觉
    :param zed0: 无人机相机右目视觉
    :type  ndarrays
    :param id: 目标任务的id
    :type  int

    :return: 人脸的位置，相当于世界坐标系
    :rtype: [x, y, z] with respect to the world
            return None for no faces in the picture
    """
    faces1, angles1, processed_img1, M1 = face_selector(zed1)
    faces0, angles0, processed_img0, M0 = face_selector(zed0)
    people = []
    max_sim = -1
    max_id = -1
    if faces1 is None or faces0 is None:
        return None
    else:
        for i in range(0, min(len(faces1), len(faces0))):
            x, y, w, h = faces1[i]
            x0, y0, w0, h0 = faces0[i]
            print(faces0)
            print(faces1)
            face1 = processed_img1[y:y+h, x:x+w, :]
            # face1, M = rotate_image(face1, -45)
            h, w = face1.shape[:2]
            face0 = processed_img0[y:y+h, x:x+w, :]
            face0, M = rotate_image(face0, 135)
            display_image(face0)
            # people.append(resize_image(blanking(person1), w, h))
            # people.append(resize_image(blanking(person2), w, h))
            # people.append(resize_image(blanking(person3), w, h))
            # people.append(resize_image(blanking(person4), w, h))
            # people.append(resize_image(blanking(person5), w, h))
            # people.append(resize_image(blanking(person6), w, h))
 
            people.append(resize_image(person[0], w, h))
            people.append(resize_image(person[1], w, h))
            people.append(resize_image(person[2], w, h))
            people.append(resize_image(person[3], w, h))
            people.append(resize_image(person[4], w, h))
            people.append(resize_image(person[5], w, h))
            
            for j in range(0, len(people)):
                sim = compare_ssim(blanking(face1), people[j], multichannel=True)
                print(sim)
                if sim > max_sim:
                    max_sim = sim
                    max_id = j + 1
            print(max_id)
            if max_id == id:
                coord1 = rotate_image_reverse([x+w/2, y+h/2], M1)
                coord0 = rotate_image_reverse([x0+w0/2, y0+h0/2], M0)
                # return reprojectionTo3D(clientID, coord1, coord0)
    return None

def linear_checking_left(x0, y0, x1, y1, xp, yp):
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    D = A * xp + B * yp + C
    if D < 0:
        return True
    return False

def blanking(img):
    h, w = img.shape[:2]
    for i in range(0, h):
        for j in range(0, w): 
            if linear_checking_left(h/2, 0, 0, w/2, i, j)==True or linear_checking_left(h/2, 0, h, w/2, i, j)==False \
                or linear_checking_left(0, w/2, h/2, w, i, j)==True or linear_checking_left(h/2, w, h, w/2, i, j)==True:
                img[i][j] = 0
    
    return img


if __name__ == '__main__':
    zed1 = cv2.imread('../fuck_l.jpeg')
    zed0 = cv2.imread('../fuck_r.jpeg')
    resNet(zed1, zed0, 0, 4)
    # zed = cv2.cvtColor(zed, cv2.COLOR_BGR2GRAY)
    # zed, _ = rotate_image(zed, 180)
    # h, w = zed.shape[:2]
    # zed1 = resize_image(zed, w * 2, h * 2)
    # faces = face_detector_dlib(zed1)
    
    # for index, face in enumerate(faces):
    #     checkin = True
    #     left = face.left()
    #     top = face.top()
    #     right = face.right()
    #     bottom = face.bottom()
    #     cv2.rectangle(zed, (int(left/2), int(top/2)), (int(right/2), int(bottom/2)), (0, 255, 0), 1) 

    # display_image(zed)
    # zed1 = cv2.cvtColor(zed1, cv2.COLOR_BGR2GRAY)

    # people = []
    # h, w = zed1.shape[:2]
    # people.append(resize_image(blanking(person1), w, h))
    # people.append(resize_image(blanking(person2), w, h))
    # people.append(resize_image(blanking(person3), w, h))
    # people.append(resize_image(blanking(person4), w, h))
    # people.append(resize_image(blanking(person5), w, h))
    # people.append(resize_image(blanking(person6), w, h))

    # for i in range(0, 6):
    #     sim = compare_ssim(zed1, people[i], multichannel=True)
    #     print(str(i) + ': ')
    #     print(sim) 