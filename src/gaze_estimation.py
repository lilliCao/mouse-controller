
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class ModelGazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Set instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions=extensions

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def load_model(self):
        '''
        Loading model in core
        '''
        self.plugin = IECore()
        if self.extensions:
            self.plugin.add_extension(self.extensions,self.device)
        self.exec_network = self.plugin.load_network(network=self.model, device_name=self.device)

    def predict(self, left_eye, right_eye, head_pose):
        '''
        Estimating gaze from eyes and head pose

        :left_eye: image of left eye
        :right_eye: image of right eye
        :head_pose: (y,p,r) of face
        :return: gaze: array of (y,p,r)
        '''
        left_eye_p, right_eye_p = self.preprocess_input(left_eye), self.preprocess_input(right_eye)

        self.exec_network.infer({'left_eye_image':left_eye_p, 'right_eye_image':right_eye_p, 'head_pose_angles': head_pose})

        result = self.exec_network.requests[0]

        gaze = result.outputs['gaze_vector']
        return gaze

    def preprocess_input(self, image):
        '''
        Preprocessing the input to fit the the inference engine
        '''
        b, c, h, w = 1,3,60,60
        prepo = np.copy(image)
        prepo = cv2.resize(prepo, (w,h))
        prepo = prepo.transpose((2,0,1))
        prepo = prepo.reshape(1,c,h,w)
        return prepo
