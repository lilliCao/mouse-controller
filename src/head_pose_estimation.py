from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

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

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape

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
        Estimating pose in face

        :image: face to predict
        :return: pose: array of (y,p,r)
                 preprocessed_image: image with drawn head pose
        '''
        preprocessed_input = self.preprocess_input(image)

        self.exec_network.infer({self.input_name:preprocessed_input})

        result = self.exec_network.requests[0]

        pose = np.stack((result.outputs['angle_y_fc'][0],result.outputs['angle_p_fc'][0],result.outputs['angle_r_fc'][0]), axis=1)

        #DEBUG draw (y,p,r) value in face
        preprocessed_image = self.draw_output(pose, image)

        return pose, preprocessed_image

    def draw_output(self, pose_out, image):
        '''
        Drawing head pose
        '''
        pose = pose_out[0].tolist()
        text_y = 'y= {:.2f}'.format(pose[0])
        text_p = 'p= {:.2f}'.format(pose[1])
        text_r = 'r= {:.2f}'.format(pose[2])
        cv2.putText(image, text_y, (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(image, text_p, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(image, text_r, (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
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
