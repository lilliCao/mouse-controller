# Computer Pointer Controller

This project is aim to use gaze detection model to control the mouse pointer of the computer. To do so, it takes advantage of 4 models including
  * [Face detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
  * [Landmarks detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
  * [Head pose estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html )
  * [Gaze estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html )


## Project Set Up and Installation

The project code is structured as following:

**/bin** : contains the given input video *demo.mp4*, a shorter video of 10s cut from the demo.mp4 for testing *test.mp4*

**/result** : a demo output video *demo_output.mp4*, a demo command line output *cmd_output.jpg*

**/requirements.txt** : contains a list of all necessary dependencies to run this application

**/src/** : contains 4 model classes (face_detection.py, facial_landmarks_detection.py, gaze_estimation.py and head_pose_estimation.py), the input_feeder.py (modified), mouse_controller.py (not modified) and the main.py.

**README.md** : contains project summary

The models themselves are downloaded corresponding to the link in the project description. In my case I put them into the default folder of ir of openvino which is **~/openvino_models/ir/intel/**

The project is tested on my local machine with installed **openvino_2020.2.120** and within the virtual environment.

```console
python3 -m venv mouse-controller-env
source mouse-controller-env/bin/activate
pip install -r requirements.txt
```

## Demo
To run the application from the project directory:
```console
cd /src
python3 main.py --model_face [face detection model path] \
--model_landmark [landmarks detection model path] \
--model_pose [head pose estimation model path] \
--model_gaze [gaze estimation model path] \
--video [video path] \
--output_path [output path] \
```
As new version of openvino will search for available extensions itself, I did not have to add extensions my own. But if you use an old version, there will be some unsupported_layers reported, please add extensions by
```console
--extensions /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
```

There are some other parameters which can be passed such as --threshold_face_detection, --device,  --extensions, --mouse_precision or --mouse_speed. In the simple case, all default values are used! Use --show_frame to show intermediate result every 5 frames

E.g.: Running the application with my project setup above on 'CPU', which results the /bin/output_video.mp4. I then removed it to  **/result/demo_output.mp4**
```console
python3 main.py --model_face ~/openvino_models/ir/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
 --model_landmark ~/openvino_models/ir/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
 --model_pose ~/openvino_models/ir/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
 --model_gaze ~/openvino_models/ir/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
 --video ../bin/demo.mp4 --output_path ../bin/
```

![Demo command line output](./result/cmd.jpg)

## Documentation

  * Please refer to the main() in main.py to have an overview about possible arguments. There is a description about each parameter in 'help'
  * The application uses synchronous inference. Face is detected from the ModelFaceDetection. If more or less than 1 face is detected, the frame is skipped for further process. Detected face is then passed to ModelLandmarksDetection and ModelHeadPoseEstimation to get the eyes and head pose, which are then passed to ModelGazeEstimation to have the final gaze_vector. (x,y) from gaze_vector is then passed to MouseController. The frame is drawn with the head pose (y,p,r), gaze vector (x,y,z), gaze vector as a arrow line from center eye, bounding box around face, eyes and point at nose and left and right corner of mouth (see /bin/demo_output.mp4). Each model class draws its own output to the frame in draw_output function
  * As I noticed the movement of mouse isn't enough accurate and also not work so good if --show_frame flag is set, I temporally comment out the call of move() function in main. I also modified the MouseController to move mouse to center at initialization
  ```python
  #TODO Uncomment the following line to control mouse movement with pyautogui
  #mouse_controller.move(gaze[0][0], gaze[0][1])
  ```
  * The intermediate results can be show if --show_frame is passed and can be stored if the '--output_path' is given.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
