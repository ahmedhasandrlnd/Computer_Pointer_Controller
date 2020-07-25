import math
from openvino.inference_engine import IECore, IENetwork
import cv2
import logging

class GazeEstimationModel():
    """
    Class for the Gaze Estimation Model.
    """
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Gaze Estimation Model class object
        """
        self.model_structure = model_path
        self.model_weights = model_path.replace('.xml', '.bin')
        self.device_name = device
        self.threshold = threshold
        self.logger = logging.getLogger('fd')
        self.model_name = 'Basic Model'
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.network = None

        try:
            self.core = IECore()
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            self.logger.error("Error While Initilizing" + str(self.model_name) + str(e))
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.model_name = 'Face Detection Model'
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [o for o in self.model.outputs.keys()]

    def load_model(self):
        """
        This method with load model using IECore object
        return loaded model
        """
        try:
            self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error While Loading"+str(self.model_name)+str(e))

    def predict(self, left_eye_image, right_eye_image, hpe_cords, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        try:
            left_eye_image = self.preprocess_input(left_eye_image)
            right_eye_image = self.preprocess_input(right_eye_image)
            self.network.start_async(request_id, inputs={'left_eye_image': left_eye_image,
                                                         'right_eye_image': right_eye_image,
                                                         'head_pose_angles': hpe_cords})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                mouse_cord, gaze_vector = self.preprocess_output(outputs, hpe_cords)
        except Exception as e:
            self.logger.error("Error While Prediction in Gaze Estimation Model" + str(e))

        return mouse_cord, gaze_vector

    def preprocess_input(self, image):
        """
        Input: image
        Return: Preprocessed image
        """
        try:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, *image.shape)
        except Exception as e:
            self.logger.error("Error While preprocessing Image in " + str(self.model_name) + str(e))

        return image

    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.network.requests[0].wait(-1)

        return status

    def preprocess_output(self, outputs, hpe_cords):
        """
        Model output is dictionary like this
        {'gaze_vector': array([[ 0.51141196,  0.12343533, -0.80407059]], dtype=float32)}
        containing Cartesian coordinates of gaze direction vector
        """
        gaze_vector = outputs[self.output_name[0]][0]
        mouse_cord = (0, 0)

        try:
            angle_r_fc = hpe_cords[2]
            sin_r = math.sin(angle_r_fc * math.pi / 180.0)
            cos_r = math.cos(angle_r_fc * math.pi / 180.0)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_cord = (x, y)
        except Exception as e:
            self.logger.error("Error While preprocessing output in Gaze Estimation Model" + str(e))

        return mouse_cord, gaze_vector