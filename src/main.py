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

    # initialize models
    mouse_controller=MouseController('high', 'slow')
    face_detector= ModelFaceDetection(model_name=model_face, device=device, extensions=extensions, threshold=face_confidence)
    landmark_detector= ModelLandmarksDetection(model_name=model_landmark)
    pose_estimator=ModelHeadPoseEstimation(model_name=model_pose)
    gaze_estimator=ModelGazeEstimation(model_name=model_gaze)

    face_detector.load_model()
    landmark_detector.load_model()
    pose_estimator.load_model()
    gaze_estimator.load_model()

    # get input
    feed=InputFeeder(input_type='video', input_file=video_file)
    feed.load_data()
    initial_w = int(feed.getCap().get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.getCap().get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.getCap().get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.getCap().get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    print("Video size = {}x{}".format(initial_h, initial_w))

    for batch in feed.next_batch():
        face, image = face_detector.predict(batch)
        out_video.write(image.copy())
        pose, _= pose_estimator.predict(face.copy())
        eyes, _= landmark_detector.predict(face.copy())
        gaze = gaze_estimator.predict(eyes[0], eyes[1], pose)
        mouse_controller.move(gaze[0][0], gaze[0][1])
        print (gaze)

    out_video.release()
    feed.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_face', required=True)
    parser.add_argument('--model_landmark', required=True)
    parser.add_argument('--model_pose', required=True)
    parser.add_argument('--model_gaze', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--threshold_face_detection', default=0.5)
    parser.add_argument('--video', default=None)
    parser.add_argument('--extensions', default=None)
    parser.add_argument('--output_path', default='/results')

    args=parser.parse_args()

    main(args)
