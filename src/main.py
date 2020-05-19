from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from input_feeder import InputFeeder
from mouse_controller import MouseController
import argparse
import cv2
import os
import numpy as np
import time

SUPPORTED_VIDEO_FORMAT = [".mp4"]

def support_video_format(video):
    '''
    Check if the given input format is supported
    '''
    for v in SUPPORTED_VIDEO_FORMAT:
        if video.endswith(v):
            return True
    return False

def running_time_report(face_time,landmark_time,pose_time,gaze_time):
    print('...Reporting running time')
    print('......Face detection model')
    print('.........Preprocessing   = {} ms'.format(face_time[0]))
    print('.........Inference time  = {} ms'.format(face_time[1]))
    print('.........Postprocessing  = {} ms'.format(face_time[2]))
    print('......Landmarks detection model')
    print('.........Preprocessing   = {} ms'.format(landmark_time[0]))
    print('.........Inference time  = {} ms'.format(landmark_time[1]))
    print('.........Postprocessing  = {} ms'.format(landmark_time[2]))
    print('......Head pose estimation model')
    print('.........Preprocessing   = {} ms'.format(pose_time[0]))
    print('.........Inference time  = {} ms'.format(pose_time[1]))
    print('.........Postprocessing  = {} ms'.format(pose_time[2]))
    print('......Gaze estimation model')
    print('.........Preprocessing   = {} ms'.format(gaze_time[0]))
    print('.........Inference time  = {} ms'.format(gaze_time[1]))
    print('.........Postprocessing  = {} ms'.format(gaze_time[2]))

def main(args):
    # get all arguments
    model_face=args.model_face
    model_landmark=args.model_landmark
    model_pose=args.model_pose
    model_gaze=args.model_gaze
    device=args.device
    extensions=args.extensions
    video_file=args.video
    output_path=args.output_path
    face_confidence=args.threshold_face_detection
    precision=args.mouse_precision
    speed=args.mouse_speed
    show_frame=args.show_frame

    # initialize models
    print('Initializing models')
    start = time.time()
    face_detector= ModelFaceDetection(model_name=model_face, device=device, extensions=extensions, threshold=face_confidence)
    face_detector.load_model()
    print ('...Successfully loading face detection model in {:.2f} ms'.format(time.time() -start))
    start = time.time()
    landmark_detector= ModelLandmarksDetection(model_name=model_landmark)
    landmark_detector.load_model()
    print ('...Successfully loading landmarks detection model in {:.2f} ms'.format(time.time() -start))
    start = time.time()
    pose_estimator=ModelHeadPoseEstimation(model_name=model_pose)
    pose_estimator.load_model()
    print ('...Successfully loading head pose estimation model in {:.2f} ms'.format(time.time() -start))
    start = time.time()
    gaze_estimator=ModelGazeEstimation(model_name=model_gaze)
    gaze_estimator.load_model()
    print ('...Successfully loading gaze estimation model in {:.2f} ms'.format(time.time() -start))

    # get input
    print('Getting input data')
    input_type = 'video'
    if video_file=='cam':
        input_type = 'cam'
    elif not support_video_format(video_file):
        print ('Unsupported input format! Please use only video file or cam as input')
        exit(1)

    feed=InputFeeder(input_type=input_type, input_file=video_file)
    feed.load_data()
    initial_w = int(feed.getCap().get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.getCap().get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.getCap().get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.getCap().get(cv2.CAP_PROP_FPS))
    if output_path:
        out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    print('...Video size hxw= {}x{}'.format(initial_h, initial_w))

    # mouse controller
    print('Initializing mouse controller')
    if precision in ['high', 'low', 'medium'] and speed in ['fast', 'slow', 'medium']:
        center = (initial_w/2, initial_h/2)
        mouse_controller=MouseController(precision, speed, center)
    else:
        print('Please setup mouse precision and speed correctly!')
        exit(1)

    count = 0
    print('Looping through all the frame and doing inference')
    for batch in feed.next_batch():
        count = count + 1
        print (count)
        face, coord, image = face_detector.predict(batch)
        if face is None:
            print('...There might be no face or more than 1 face detected. Skip this frame')
            continue
        pose, image = pose_estimator.predict(face.copy(), image)
        eyes, eyes_center, image = landmark_detector.predict(face.copy(), coord, image)
        gaze, image = gaze_estimator.predict(eyes[0], eyes[1], pose, eyes_center, image)
        if output_path:
            out_video.write(image)
        if show_frame and (count % 5==0):
            # show intermediate result every 5 frames
            cv2.imshow('frame'.format(count), image)
            # Press Q on keyboard to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #TODO comment the following to deactivate mouse movement
        #if want to focus more on the intermediate result!
        if count%10==0:
            # pyautogui.moveRel blocking 0.1s -> blocking inference -> move only every 10 frames
            mouse_controller.move(gaze[0][0], gaze[0][1])

    if output_path:
        print('Finished inference and successfully stored output to ', os.path.join(output_path, 'output_video.mp4'))
    else:
        print('Finished inference')

    running_time_report(face_detector.get_time(), landmark_detector.get_time(), pose_estimator.get_time(), gaze_estimator.get_time())

    print('Releasing resources')
    if output_path:
        out_video.release()
    feed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_face', required=True, type=str, help='Path to model of face detection')
    parser.add_argument('--model_landmark', required=True, type=str, help='Path to model of landmarks detection')
    parser.add_argument('--model_pose', required=True, type=str, help='Path to model of pose estimation')
    parser.add_argument('--model_gaze', required=True, type=str, help='Path to model of gaze estimation')
    parser.add_argument('--device', default='CPU', type=str,
                                    help='Specify the target device to infer on: '
                                         'CPU, GPU, FPGA or MYRIAD is acceptable. Sample '
                                         'will look for a suitable plugin for device '
                                         'specified (CPU by default)')
    parser.add_argument('--threshold_face_detection', default=0.5, type=float,
                                            help='Probability threshold for detections filtering of face detection'
                                                 '(0.5 by default)')
    parser.add_argument('--video', default=None, type=str,
                                   help='Path to video file or cam if using camera')
    parser.add_argument('--extensions', default=None, type=str,
                                        help='MKLDNN (CPU)-targeted custom layers.'
                                             'Absolute path to a shared library with the'
                                             'kernels impl.')
    parser.add_argument('--output_path', default='None',
                                         help='Path to write output video (None by default means no intermediate results should be stored)')
    parser.add_argument('--show_frame', default=False, action='store_true',
                                         help='Flag to show intermediate results (False by default)')
    parser.add_argument('--mouse_precision', default='high', help='Mouse movement precision. Please pass high, low or medium')
    parser.add_argument('--mouse_speed', default='medium', help='Mouse movement speed. Please pass fast, slow or medium')
    args=parser.parse_args()

    main(args)
