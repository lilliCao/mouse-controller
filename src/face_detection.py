from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import time

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

        self.preprocessing_time=0
        self.postprocessing_time=0
        self.inference_time=0

        self.plugin = IECore()

        try:
            self.model=self.plugin.read_network(self.model_structure, self.model_weights)
        except AttributeError:
            # old openvino has no method IECore,read_network()
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network for face detection. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Loading model in core
        '''
        if self.extensions:
            self.plugin.add_extension(self.extensions,self.device)
        self.check_model()
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device)

    def check_model(self):
        '''
        Checking for unsupported layers
        '''
        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def predict(self, image):
        '''
        Detecting face in image

        :image: frame to predict
        :return: face: cropped face
                 x,y: start coordinate of cropped face
                 preprocessed_image: image with drawn bounding box
        '''
        start = time.time()
        preprocessed_input = self.preprocess_input(image)
        self.preprocessing_time = self.preprocessing_time + (time.time() -start)

        start = time.time()
        self.exec_network.infer({self.input_name:preprocessed_input})
        self.inference_time = self.inference_time + (time.time() -start)

        start = time.time()
        result = self.exec_network.requests[0]

        coords = self.preprocess_output(result.outputs[self.output_name])
        height, width = image.shape[:2]
        for coord in coords:
            coord[0] = coord[0] * width
            coord[1] = coord[1] * height
            coord[2] = coord[2] * width
            coord[3] = coord[3] * height

        # Detect more than 1 face -> don't process further
        if len(coords)!=1:
            preprocessed_image = self.draw_output(coords, image)
            return None, (None,None), preprocessed_image

        # convert np.float32 to int in order to cut the face from image
        coords_float = coords[0].tolist()
        x,y,x_end,y_end = int(coords_float[0]), int(coords_float[1]),int(coords_float[2]),int(coords_float[3])
        face = image[y:y_end,x:x_end].copy()
        preprocessed_image = self.draw_output(coords, image)

        self.postprocessing_time = self.postprocessing_time + (time.time() -start)
        return face, (x,y), preprocessed_image

    def draw_output(self, coords, image):
        '''
        Drawing rectangles around detected faces
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

    def get_time(self):
        return self.preprocessing_time, self.inference_time, self.postprocessing_time
