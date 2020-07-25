from openvino.inference_engine import IECore, IENetwork
import cv2
import logging

class LandmarkDetectionModel():
    """
    This is a class for the operation of Landmark Detection Model
    """
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Landmark Detection Model class object
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

        self.model_name = 'Landmark Detection Model'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))

    def load_model(self):
        """
        This method with load model using IECore object
        return loaded model
        """
        try:
            self.network = self.core.load_network(network=self.model, device_name=self.device_name, num_requests=1)
        except Exception as e:
            self.logger.error("Error While Loading"+str(self.model_name)+str(e))

    def predict(self, image, request_id=0):
        """
        This method will take image as a input and
        does all the preprocessing, postprocessing
        """
        left_eye_image, right_eye_image, eye_cords = [], [], []

        try:
            p_image = self.preprocess_img(image)
            self.network.start_async(request_id, inputs={self.input_name: p_image})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs[self.output_name]
                left_eye_image, right_eye_image, eye_cords = self.preprocess_output(outputs, image)
        except Exception as e:
            self.logger.error("Error While making prediction in Landmark Detection Model" + str(e))

        return left_eye_image, right_eye_image, eye_cords

    def preprocess_img(self, image):
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

    def preprocess_output(self, outputs, image):
        """
        Return the left and right eyes coordinates
        """
        h = image.shape[0]
        w = image.shape[1]
        left_eye_image, right_eye_image, eye_cords = [], [], []
        try:
            outputs = outputs[0]

            left_eye_xmin = int(outputs[0][0][0] * w) - 10
            left_eye_ymin = int(outputs[1][0][0] * h) - 10
            right_eye_xmin = int(outputs[2][0][0] * w) - 10
            right_eye_ymin = int(outputs[3][0][0] * h) - 10

            left_eye_xmax = int(outputs[0][0][0] * w) + 10
            left_eye_ymax = int(outputs[1][0][0] * h) + 10
            right_eye_xmax = int(outputs[2][0][0] * w) + 10
            right_eye_ymax = int(outputs[3][0][0] * h) + 10

            left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
            right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]

            eye_cords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax],
                         [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]

        except Exception as e:
            self.logger.error("Error While drawing bounding boxes on image in Landmark Detection Model" + str(e))

        return left_eye_image, right_eye_image, eye_cords