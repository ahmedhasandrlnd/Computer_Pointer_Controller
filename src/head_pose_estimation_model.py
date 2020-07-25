from openvino.inference_engine import IECore, IENetwork
import cv2
import logging

class HeadPoseEstimationModel():
    """
    This is a class for the operation of Head Pose Estimation Model
    """
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        """
        This will initiate Head Pose Estimation Model class object
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

        self.model_name = 'Head Pose Estimation Model'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

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
        try:
            p_image = self.preprocess_input(image)
            self.network.start_async(request_id, inputs={self.input_name: p_image})
            if self.wait() == 0:
                outputs = self.network.requests[0].outputs
                f_output = self.preprocess_output(outputs)
        except Exception as e:
            self.logger.error("Error While prediction in Head Pose Estimation Model" + str(e))

        return f_output

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

    def preprocess_output(self, outputs):
        """
        Model output is a dictionary having below three arguments:
        """
        final_output = []
        try:
            final_output.append(outputs['angle_y_fc'][0][0])
            final_output.append(outputs['angle_p_fc'][0][0])
            final_output.append(outputs['angle_r_fc'][0][0])
        except Exception as e:
            self.logger.error("Error While preprocessing output in Head Pose Estimation Model" + str(e))

        return final_output