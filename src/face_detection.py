
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class ModelFaceDetection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        Set instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions=extensions
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Loading model in core
        '''
        self.plugin = IECore()
        if self.extensions:
            self.plugin.add_extension(self.extensions,self.device)
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device)

    def predict(self, image):
        '''
        Detecting face in image

        :image: frame to predict
        :return: face: cropped face
                 preprocessed_image: image with drawn bounding box
        '''
        preprocessed_input = self.preprocess_input(image)

        self.exec_network.infer({self.input_name:preprocessed_input})

        result = self.exec_network.requests[0]

        coords = self.preprocess_output(result.outputs[self.output_name])
        height, width = image.shape[:2]
        for coord in coords:
            coord[0] = coord[0] * width
            coord[1] = coord[1] * height
            coord[2] = coord[2] * width
            coord[3] = coord[3] * height

        # Detect more than 1 face -> don't process further
        if len(coords)>1:
            preprocessed_image = self.draw_output(coords, image)
            return None, preprocessed_image

        # convert np.float32 to int
        coords_float = coords[0].tolist()
        x,y,x_end,y_end = int(coords_float[0]), int(coords_float[1]),int(coords_float[2]),int(coords_float[3])
        face = image[y:y_end,x:x_end].copy()
        preprocessed_image = self.draw_output(coords, image)

        return face, preprocessed_image

    def draw_output(self, coords, image):
        '''
        Drawing rectangles around detected people
        '''
        for coord in coords:
            (startX, startY, endX, endY) = coord
            cv2.rectangle(image, (startX, startY), (endX, endY), (255,0,0), 2)
        return image

    def preprocess_input(self, image):
        '''
        Preprocessing the input to fit the the inference engine
        '''
        b, c, h, w = self.input_shape
        prepo = np.copy(image)
        prepo = cv2.resize(prepo, (w,h))
        prepo = prepo.transpose((2,0,1))
        prepo = prepo.reshape(self.input_shape)
        return prepo

    def preprocess_output(self, outputs):
        '''
        Processing the output to get the bounding box
        '''
        coords = []
        for i in np.arange(0, outputs.shape[2]):
            box = outputs[0, 0, i, 3:7]
            confidence = outputs[0, 0, i, 2]
            if confidence > self.threshold:
                coords.append(box)
        return coords
