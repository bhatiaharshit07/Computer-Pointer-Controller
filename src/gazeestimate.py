import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import math


class gaze_estimate:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.plugin = IECore()
        self.left_eye_img = None
        self.right_eye_img = None

        try:
            self.model = self.plugin.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        self.plugin = IECore()
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, head_pose_output):
        self.left_eye_img, self.right_eye_img = self.preprocess_input(left_eye_image, right_eye_image)
        input_dict = {'left_eye_image': self.left_eye_img, 'right_eye_image': self.right_eye_img, 'head_pose_angles': head_pose_output}
        self.results = self.net.infer(input_dict)
        self.mouse_coords, self.gaze_vector = self.preprocess_output(self.results, head_pose_output)
        return self.mouse_coords, self.gaze_vector

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_eye_image, right_eye_image):
        left_eye_image = cv2.resize(left_eye_image, (60, 60))
        left_eye_image = left_eye_image.transpose((2, 0, 1))
        left_eye_image = left_eye_image.reshape(1, *left_eye_image.shape)
        right_eye_image = cv2.resize(right_eye_image, (60, 60))
        right_eye_image = right_eye_image.transpose((2, 0, 1))
        right_eye_image = right_eye_image.reshape(1, *right_eye_image.shape)
        return left_eye_image, right_eye_image

    def preprocess_output(self, outputs, head_pose_estimation_output):
        rollAlign = head_pose_estimation_output[2]
        outs = outputs[self.output_name][0]
        cos_angle = math.cos(rollAlign * math.pi / 180)
        sin_angle = math.sin(rollAlign * math.pi / 180)
        x_temp = outs[0] * cos_angle + outs[1] * sin_angle
        y_temp = outs[1] * cos_angle - outs[0] * sin_angle
        return (x_temp, y_temp), outputs
