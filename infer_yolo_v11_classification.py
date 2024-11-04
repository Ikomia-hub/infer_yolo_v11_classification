from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_yolo_v11_classification.infer_yolo_v11_classification_process import InferYoloV11ClassificationFactory
        return InferYoloV11ClassificationFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_yolo_v11_classification.infer_yolo_v11_classification_widget import InferYoloV11ClassificationWidgetFactory
        return InferYoloV11ClassificationWidgetFactory()
