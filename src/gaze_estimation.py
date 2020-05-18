
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

        self.plugin = IECore()

        try:
            self.model=self.plugin.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def load_model(self):
        '''
        Loading model in core
        '''
        self.check_model()
        if self.extensions:
            self.plugin.add_extension(self.extensions,self.device)
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

    def predict(self, left_eye, right_eye, head_pose, eyes_center, origin_image):
        '''
        Estimating gaze from eyes and head pose

        :left_eye: image of left eye
        :right_eye: image of right eye
        :head_pose: (y,p,r) of face
        :origin_image: image to visualize output
        :return: gaze: array of (x,y,z)
                 preprocessed_image: image with drawn gaze vector
        '''
        left_eye_p, right_eye_p = self.preprocess_input(left_eye), self.preprocess_input(right_eye)

        self.exec_network.infer({'left_eye_image':left_eye_p, 'right_eye_image':right_eye_p, 'head_pose_angles': head_pose})

        result = self.exec_network.requests[0]

        gaze = result.outputs['gaze_vector']

        #DEBUG draw (x,y,z) value of gaze_vector
        preprocessed_image = self.draw_output(gaze, eyes_center, origin_image)

        return gaze, preprocessed_image

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

    def draw_output(self, gaze_vector, eyes_center, image):
        '''
        Drawing gaze_vector
        '''
        gaze = gaze_vector[0].tolist()
        text = 'gaze x= {:.1f}, y= {:.1f}, z= {:.1f}'.format(gaze[0], gaze[1], gaze[2])
        cv2.putText(image, text, (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

        norm = 200
        left_eye_center = eyes_center[0],eyes_center[1]
        left_gaze_point = (int(gaze[0]*norm+left_eye_center[0]), int(gaze[1]*norm*(-1)+left_eye_center[1]))
        right_eye_center = eyes_center[2],eyes_center[3]
        right_gaze_point = (int(gaze[0]*norm+right_eye_center[0]), int(gaze[1]*norm*(-1)+right_eye_center[1]))

        cv2.arrowedLine(image, left_eye_center , left_gaze_point, (0,0,255), 2)
        cv2.arrowedLine(image, right_eye_center , right_gaze_point, (0,0,255), 2)

        return image
