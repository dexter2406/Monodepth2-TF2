import os
import numpy as np


class DepthEstimationModel(object):
    encoder_models = ['encoder_res18_singlet', None]
    decoder_models = ['depth_decoder_singlet', None]

    def __init__(self, weights_dir:str=None, encoder_name:str ='resnet18', decoder_name:str = 'depth_decoder',
                 encoder_net=None, decoder_net=None, esimate_func=None, single_mode=True):

        self.single_mode = single_mode
        self.weights_dir = weights_dir
        if self.single_mode:
            self.encoder_path = os.path.join(self.weights_dir, self.encoder_models[0])
        else:
            raise NotImplementedError
            # self.pathpath = ''.join([self.weights_dir, encoder[1]])
        self.decoder_path = os.path.join(self.weights_dir, self.decoder_models[0])
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

        self.original_size: tuple = None

        self.esimate_func = esimate_func
        self.features = {}
        self.disp: np.ndarray = None

        self.scaling: float = 1.

    def estimate_depth(self, image):
        return self.esimate_func(image, model=self)

    def clear_results(self):
        self.features = {}
        self.disp = None
