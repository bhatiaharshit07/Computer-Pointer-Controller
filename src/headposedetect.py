import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class head_pose_detect:

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
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
    
    def load_model(self):
        self.plugin = IECore()
        self.net = self.plugin.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        self.img = self.preprocess_input(image)
        input_dict = {self.input_name:self.img}
        results = self.net.infer(input_dict)
        self.Output = self.preprocess_output(results)
        return self.Output

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        outs = []
        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])
        return outs
