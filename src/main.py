import cv2
import os
import numpy as np
import math
import time
from facedetect import face_detect
from gazeestimate import gaze_estimate
from landmarkdetect import landmark_detect
from headposedetect import head_pose_detect
from mousecontroller import mouse_controller
from input_feeder import InputFeeder
from argparse import ArgumentParser

def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file")
    parser.add_argument("-ld", "--facial_landmarks_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, 
                        default=None, 
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-flag", "--informations_flag", required=False, nargs='+',
                        default=[],
                        help="Example: --flag fd ld hp ge (Seperate each flag by space)"
                             "to see the visualization of different model outputs of each frame,"
                             "fd for Face Detection Model, ld for Facial Landmark Detection Model"
                             "hp for Head Pose Estimation Model, ge for Gaze Estimation Model.")
    return parser

def main():
    args = build_argparser().parse_args()
    input_file = args.input
    info_flag = args.informations_flag
    if input_file == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file):
            print("ERROR: INPUT PATH IS NOT VALID")
            exit(1)
        input_feeder = InputFeeder("video", input_file)
    
    model_dict = {'Face_detection_model': args.face_detection_model, 'Facial_landmarks_detection_model': args.facial_landmarks_model, 'head_pose_estimation_model': args.head_pose_model, 'gaze_estimation_model': args.gaze_estimation_model}
    Face_Detect = face_detect(model_name=model_dict['Face_detection_model'], device=args.device, threshold=args.prob_threshold, extensions=args.cpu_extension)
    Landmark_Detect = landmark_detect(model_name=model_dict['Facial_landmarks_detection_model'], device=args.device, extensions=args.cpu_extension)
    Gaze_Estimate = gaze_estimate(model_name=model_dict['gaze_estimation_model'], device=args.device, extensions=args.cpu_extension)
    Head_Pose_Detect = head_pose_detect(model_name=model_dict['head_pose_estimation_model'], device=args.device, extensions=args.cpu_extension)
    Mouse_Controller = mouse_controller('medium', 'fast')

    input_feeder.load_data()
    start_time = time.time()
    Face_Detect.load_model()
    Landmark_Detect.load_model()
    Head_Pose_Detect.load_model()
    Gaze_Estimate.load_model()
    total_models_load_time = time.time() - start_time    

    counter = 0
    start_inf_time = time.time()
    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        pressed_key = cv2.waitKey(60)
        counter = counter + 1
        face_coords, crop_face = Face_Detect.predict(frame.copy())
        if face_coords == 0:
            continue
        head_pose_detect_output = Head_Pose_Detect.predict(crop_face)
        left_eye_image, right_eye_image, eye_coord = Landmark_Detect.predict(crop_face)
        mouse_coordinate, gaze_vector = Gaze_Estimate.predict(left_eye_image, right_eye_image, head_pose_detect_output)

        if len(info_flag) != 0:
            preview_window = frame.copy()
            if 'fd' in info_flag:
                if len(info_flag) != 1:
                    preview_window = crop_face
                else:
                    cv2.rectangle(preview_window, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0, 150, 0), 3)
            if 'ld' in info_flag:
                if not 'fd' in info_flag:
                    preview_window = crop_face.copy()
                cv2.rectangle(preview_window, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]), (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]), (150, 0, 150))
            if 'hp' in info_flag:
                cv2.putText(preview_window, "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(head_pose_detect_output[0], head_pose_detect_output[1], head_pose_detect_output[2]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
            if 'ge' in info_flag:
                yaw = head_pose_detect_output[0]
                pitch = head_pose_detect_output[1]
                roll = head_pose_detect_output[2]
                focal_length = 950.0
                scale = 50
                center_of_face = (crop_face.shape[1] / 2, crop_face.shape[0] / 2, 0)
                if 'fd' in info_flag or 'ld' in info_flag:
                    draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                else:
                    draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)

        if len(info_flag) != 0:
            img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
        else:
            img_hor = cv2.resize(frame, (500, 500))

        cv2.imshow('Visualization', img_hor)
        Mouse_Controller.move(mouse_coordinate[0], mouse_coordinate[1])

        if pressed_key == 27:
            print("exit key is pressed..")
            break

    inference_time = time.time() - start_inf_time
    fps = int(counter) / inference_time

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.txt'), 'w') as f:
        f.write("Inference Time: " + str(inference_time) + '\n')
        f.write("FPS: " + str(fps) + '\n')
        f.write("Model Loading Time: " + str(total_models_load_time) + '\n')

    input_feeder.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()