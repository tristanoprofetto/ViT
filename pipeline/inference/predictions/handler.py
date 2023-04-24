import torch
from ts.torch_handler.base_handler import BaseHandler


class ViTModelHandler(BaseHandler):
    """
    Custom model handler for ViT classifier
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0


    def initialize(self, context):
        self._context = context
        self.initialized = True
        self.model = torch.load()
        pass

    
    def preprocess(self):
        pass


    def postprocess(self):
        pass


    def inference(self):
        pass


    def handle(self):
        pass