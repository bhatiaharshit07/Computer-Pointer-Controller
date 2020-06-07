import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class landmark_detect:

    def __init__(self, model_name, device, extensions=None):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions
        self.plugin = IECore()
        self.img = None

        try:
            self.model = self.plugin.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        self.plugin = IECore()
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
    
        self.img = self.preprocess_input(image)
        input_dict = {self.input_name:self.img}
        self.results = self.net.infer(input_dict)
        
        self.output = self.preprocess_output(self.results, image)

        left_eye_x_min = self.output['left_eye_x'] - 5
        left_eye_x_max = self.output['left_eye_x'] + 5
        left_eye_y_min = self.output['left_eye_y'] - 5
        left_eye_y_max = self.output['left_eye_y'] + 5

        right_eye_x_min = self.output['right_eye_x'] - 5
        right_eye_x_max = self.output['right_eye_x'] + 5
        right_eye_y_min = self.output['right_eye_y'] - 5
        right_eye_y_max = self.output['right_eye_y'] + 5

        self.eye_coords = [[left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max],
                          [right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max]]
        left_eye_image = image[left_eye_x_min:left_eye_x_max, left_eye_y_min:left_eye_y_max]
        right_eye_image = image[right_eye_x_min:right_eye_x_max, right_eye_y_min:right_eye_y_max]

        return left_eye_image, right_eye_image, self.eye_coords

    def check_model(self):
        pass

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs, image):
        outs = outputs[self.output_name][0]
        left_eye_x = int(outs[0] * image.shape[1])
        left_eye_y = int(outs[1] * image.shape[0])
        right_eye_x = int(outs[2] * image.shape[1])
        right_eye_y = int(outs[3] * image.shape[0])

        return {'left_eye_x': left_eye_x, 'left_eye_y': left_eye_y,
                'right_eye_x': right_eye_x, 'right_eye_y': right_eye_y}
