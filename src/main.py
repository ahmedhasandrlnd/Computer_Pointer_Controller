import cv2
import os
import logging
import time
import numpy as np
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection_model import FaceDetectionModel
from landmark_detection_model import LandmarkDetectionModel
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel

def build_argparser():
    """
    parse commandline argument
    return ArgumentParser object
    """
    parser = ArgumentParser()
    parser.add_argument("-mf", "--faceDetectionModel", type=str, required=True,
                        help="Specify path of xml file of face detection model")
    parser.add_argument("-ml", "--landmarkRegressionModel", type=str, required=True,
                        help="Specify path of xml file of landmark regression model")
    parser.add_argument("-mh", "--headPoseEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Head Pose Estimation model")
    parser.add_argument("-mg", "--gazeEstimationModel", type=str, required=True,
                        help="Specify path of xml file of Gaze Estimation model")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Specify path of input Video file or cam for webcam")
    parser.add_argument("-f", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify flag from ff, fl, fh, fg like -flags ff fl(Space separated if multiple values)"
                             "ff for faceDetectionModel, fl for landmarkRegressionModel"
                             "fh for headPoseEstimationModel, fg for gazeEstimationModel")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Specify probability threshold for face detection model")
    parser.add_argument("-d", "--device", required=False, type=str, default='CPU',
                        help="Specify Device for inference"
                             "It can be CPU, GPU, FPGU, MYRID")
    parser.add_argument("-o", '--output_path', default='/results/', type=str)
    parser.add_argument("-b","--benchmark", required=False, default=False, action="store_true" , help="Choose for benchmarking mode")

    return parser

def output_preview(
        frame, preview_flags, cropped_image, face_cords, eye_cords, pose_output, gaze_vector):
    le_mid_x, le_mid_y=int((eye_cords[0][0]+eye_cords[0][2])/2),int((eye_cords[0][1]+eye_cords[0][3])/2)
    re_mid_x, re_mid_y=int((eye_cords[1][0]+eye_cords[1][2])/2),int((eye_cords[1][1]+eye_cords[1][3])/2)
    # Face Detection output
    if 'ff' in preview_flags:
        cv2.rectangle(frame, (face_cords[0][0], face_cords[0][1]), (face_cords[0][2], face_cords[0][3]),
                      (255, 0, 0), 3)
    # Landmark output
    if 'fl' in preview_flags:
        cv2.circle(cropped_image, (le_mid_x, le_mid_y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(cropped_image, (re_mid_x, re_mid_y), radius=5, color=(0, 255, 0), thickness=-1)
    # Head Position output    
    if 'fh' in preview_flags:
        cv2.putText(
            frame,
            "Pose Angles: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(
                pose_output[0], pose_output[1], pose_output[2]),
            (20, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (255, 0, 255), 2)
    # Gaze output    
    if 'fg' in preview_flags:
        cv2.putText(
            frame,
            "Gaze Cords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                gaze_vector[0], gaze_vector[1], gaze_vector[2]),
            (20, 80),
            cv2.FONT_HERSHEY_COMPLEX,
            1, (255, 0, 255), 2)

        x, y, w = int(gaze_vector[0] * 100), int(gaze_vector[1] * 100), 160
        cv2.arrowedLine(cropped_image, (le_mid_x, le_mid_y), (le_mid_x+x, le_mid_y-y), (0, 0, 255), 3)
        cv2.arrowedLine(cropped_image, (re_mid_x, re_mid_y), (re_mid_x+x, re_mid_y-y), (0, 0, 255), 3)

    return cropped_image

def main():
    args = build_argparser().parse_args()
    logger = logging.getLogger('main')
    is_benchmarking = args.benchmark
    preview_flags = args.previewFlags
    input_filename = args.input
    device_name = args.device
    prob_threshold = args.prob_threshold
    output_path = args.output_path

    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')

    else:

        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    model_paths=[args.faceDetectionModel,args.landmarkRegressionModel,args.headPoseEstimationModel,args.gazeEstimationModel]    
    for model_path in model_paths:

        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file" + str(model_path))
            exit(1)

    face_detection_model = FaceDetectionModel(args.faceDetectionModel, device_name, threshold=prob_threshold)
    landmark_detection_model = LandmarkDetectionModel(args.landmarkRegressionModel, device_name, threshold=prob_threshold)
    head_pose_estimation_model = HeadPoseEstimationModel(args.headPoseEstimationModel, device_name, threshold=prob_threshold)
    gaze_estimation_model = GazeEstimationModel(args.gazeEstimationModel, device_name, threshold=prob_threshold)

    if not is_benchmarking:
        mouse_controller = MouseController('medium', 'fast')

    # Load Mode    
    start_model_load_time = time.time()
    face_detection_model.load_model()
    landmark_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_model_load_time
    feeder.load_data()
    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.get_fps()/10),
                                (1920, 1080), True)
    frame_count = 0
    start_inference_time = time.time()

    for ret, frame in feeder.next_batch():

        if not ret:
            break

        frame_count += 1
        key = cv2.waitKey(60)
        
        try:
            # Pipeling models
            face_cords, cropped_image = face_detection_model.predict(frame)

            if type(cropped_image) == int:
                logger.warning("Unable to detect the face")

                if key == 27:
                    break
                continue

            left_eye_image, right_eye_image, eye_cords = landmark_detection_model.predict(cropped_image)
            pose_output = head_pose_estimation_model.predict(cropped_image)
            mouse_cord, gaze_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image, pose_output)

        except Exception as e:
            logger.warning("Could predict using model" + str(e) + " for frame " + str(frame_count))
            continue

        image = cv2.resize(frame, (500, 500))

        if not len(preview_flags) == 0:
            preview_frame = output_preview(
                frame, preview_flags, cropped_image, face_cords, eye_cords, pose_output, gaze_vector)
            image=cv2.resize(frame, (1000, 1000))
        cv2.imshow('preview', image)
        out_video.write(frame)

        # Move mouse
        if frame_count % 5 == 0 and not is_benchmarking:
            mouse_controller.move(mouse_cord[0], mouse_cord[1])

        if key == 27:
            break
    
    # Calculate performance        
    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    try:
        os.mkdir(output_path)
    except OSError as error:
        logger.error(error)

    with open(output_path+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(total_model_load_time) + '\n')

    # Logging output    
    logger.info('Model load time: ' + str(total_model_load_time))
    logger.info('Inference time: ' + str(total_inference_time))
    logger.info('FPS: ' + str(fps))
    logger.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()

if __name__ == '__main__':
    main()