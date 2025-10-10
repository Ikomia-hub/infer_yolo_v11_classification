import copy
import os
import torch
from ikomia import core, dataprocess, utils
from ultralytics import YOLO
from ultralytics import download


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV11ClassificationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolo11m-cls"
        self.cuda = torch.cuda.is_available()
        self.input_size = 224
        self.update = False
        self.model_weight_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "update": str(self.update),
            "model_weight_file": str(self.model_weight_file)
        }
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV11Classification(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)
        # Create parameters object
        if param is None:
            self.set_param_object(InferYoloV11ClassificationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False
        self.model_name = None
        self.repo = 'ultralytics/assets'
        self.version = 'v8.3.0'

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        self.half = True if param.cuda and torch.cuda.is_available() else False

        if param.model_weight_file:
            self.model = YOLO(param.model_weight_file)
        else:
            # Set path
            model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
            model_weights = os.path.join(str(model_folder), f'{param.model_name}.pt')

            # Download model if not exist
            if not os.path.isfile(model_weights):
                url = f'https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt'
                download(url=url, dir=model_folder, unzip=True)

            self.model = YOLO(model_weights)

        categories = list(self.model.names.values())
        self.set_names(categories)
        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # Load model
        if param.update:
            self._load_model()

        # Inference on whole image
        if self.is_whole_image_classification():
            # Run detection
            results = self.model.predict(
                src_image,
                save=False,
                imgsz=param.input_size,
                half=self.half,
                device=self.device
            )

            # Get result output (classes, confidences)
            classes_names = results[0].names
            probs = results[0].probs
            classes_idx = probs.top5

            # Map each index to the class name
            t5_class_names = [classes_names[idx] for idx in classes_idx]
            t5_confidences = probs.top5conf.detach().cpu().numpy()

            # Convert each conf to str
            confidence_str = [str(conf) for conf in t5_confidences]

            # Display results
            self.set_whole_image_results(t5_class_names, confidence_str)

        # Inference on ROIs
        else:
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                results = self.model.predict(
                    roi_img,
                    save=False,
                    imgsz=param.input_size,
                    half=self.half,
                    device=self.device
                )

                probs = results[0].probs
                classes_idx = probs.top1
                confidence = probs.top1conf.detach().cpu().numpy()
                self.add_object(obj, classes_idx, float(confidence))

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV11ClassificationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        self.info.name = "infer_yolo_v11_classification"
        self.info.short_description = "Inference with YOLOv11 image classification models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.1.0"
        self.info.min_ikomia_version = "0.15.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "AGPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://docs.ultralytics.com/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolo_v11_classification"
        self.info.original_repository = "https://github.com/ultralytics/ultralytics"
        # Keywords used for search
        self.info.keywords = "YOLO, classification, ultralytics, coco"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "CLASSIFICATION"
        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferYoloV11Classification(self.info.name, param)
