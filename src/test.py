from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
import argparse
import cv2
import os
import numpy as np



def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    output_path=args.output_path
    confidence=args.threshold

    pd= ModelFaceDetection(model_name=model, threshold=confidence)
    #pd= ModelLandmarksDetection(model_name=model)
    #pd=ModelHeadPoseEstimation(model_name=model)
    pd.load_model()

    image_flag = False
    if video_file == 'CAM':
        video_file = 0
    elif video_file.endswith('.jpg'):
        image_flag = True

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
        exit(-1)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        exit(-1)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if not image_flag:
        out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    out_image = os.path.join(output_path, 'output.jpg')
    out_image_2 = os.path.join(output_path, 'output2.jpg')
    print("Video size = {}x{}".format(initial_h, initial_w))
    try:
        while cap.isOpened():
            ret, frame=cap.read()
            print (frame.shape)
            if not ret:
                break

            #eyes, image= pd.predict(frame)
            face, image= pd.predict(frame)
            #pose, image= pd.predict(frame)
            if not image_flag:
                out_video.write(image)
            cv2.imwrite(out_image, image)
            #cv2.imwrite(out_image, eyes[0])
            #cv2.imwrite(out_image_2, eyes[1])

        cap.release()
        cv2.destroyAllWindows()
        if not image_flag:
            out_video.release()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')

    args=parser.parse_args()

    main(args)
