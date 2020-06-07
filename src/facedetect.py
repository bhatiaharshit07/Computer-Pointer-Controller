import cv2
import os
import numpy as np
from openvino.inference_engine import IENetwork, IECore
class face_detect:

    def __init__(self, model_name, device='CPU',threshold=0.5, extensions=None):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + ".xml"
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.net = None
        self.img = None
        self.plugin = IECore()
        try:
            self.model=self.plugin.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


    def load_model(self):
        self.plugin = IECore()
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        self.img = self.preprocess_input(image)
        input_dict = {self.input_name:self.img}
        results = self.net.infer(input_dict)
        self.faces_coordinates = self.preprocess_output(results, image)
        if len(self.faces_coordinates) == 0:
            print("No Face is detected, Next frame will be processed..")
            return 0, 0

        self.first_face = self.faces_coordinates[0]
        crop_face = image[self.first_face[1]:self.first_face[3], self.first_face[0]:self.first_face[2]]

        return self.first_face, crop_face

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs, image):
        coords = []
        outs = outputs[self.output_name][0][0]
        for obj in outs:
            conf = obj[2]
            if conf >= self.threshold:
                xmin = int(obj[3] * image.shape[1])
                ymin = int(obj[4] * image.shape[0])
                xmax = int(obj[5] * image.shape[1])
                ymax = int(obj[6] * image.shape[0])
                coords.append([xmin, ymin, xmax, ymax])
        return coords
