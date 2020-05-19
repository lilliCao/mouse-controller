from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2
import time

class ModelHeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Set instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions=extensions

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
            raise ValueError("Could not Initialise the network for head pose estimation. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape

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

    def predict(self, face, origin_image):
        '''
        Estimating pose in face

        :face: face to predict
        :origin_image: original image
        :return: pose: array of (y,p,r)
                 preprocessed_image: image with drawn head pose
        '''
        start = time.time()
        preprocessed_input = self.preprocess_input(face)
        self.preprocessing_time = self.preprocessing_time + (time.time() -start)

        start = time.time()
        self.exec_network.infer({self.input_name:preprocessed_input})
        self.inference_time = self.inference_time + (time.time() -start)

        start = time.time()
        result = self.exec_network.requests[0]

        pose = np.stack((result.outputs['angle_y_fc'][0],
                         result.outputs['angle_p_fc'][0],
                         result.outputs['angle_r_fc'][0]), axis=1)

        preprocessed_image = self.draw_output(pose, origin_image)
        self.postprocessing_time = self.postprocessing_time + (time.time() -start)

        return pose, preprocessed_image

    def draw_output(self, pose_out, image):
        '''
        Drawing head pose
        '''
        pose = pose_out[0].tolist()
        text = 'head pose y= {:.1f}, p= {:.1f}, r= {:.1f}'.format(pose[0], pose[1], pose[2])
        cv2.putText(image, text, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
        return image

    def preprocess_input(self, image):
        '''
        Preprocessing the input to fit the the inference engine
        '''
        b, c, h, w = self.input_shape
        prepo = np.copy(image)
        prepo = cv2.resize(prepo, (w,h))
        prepo = prepo.transpose((2,0,1))
        prepo = prepo.reshape(1,c,h,w)
        return prepo

    def get_time(self):
        return self.preprocessing_time, self.inference_time, self.postprocessing_time
