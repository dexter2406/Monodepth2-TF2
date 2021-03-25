import tensorflow as tf
import numpy as np
from collections import OrderedDict
import cv2 as cv
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import pickle
import time
from collections import defaultdict

from src.DepthEstimationModel import DepthEstimationModel


def prepare_image(image=None, feed_width=640, feed_height=192, batch_dim=True, as_tensor=True,
                  channel_first=False, type = 'ImageNet'):
    if image is None:
        image_path = 'assets/test_image.jpg'
        image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    if max(image.shape[:2]) > max(feed_width, feed_height):
        interp_op = cv.INTER_AREA
    else:
        interp_op = cv.INTER_CUBIC
    input_data = cv.resize(image, (feed_width, feed_height), interp_op)
    input_data = input_data.astype(np.float32) / 255.
    if batch_dim is False and (as_tensor or channel_first):
        warnings.warn('Since batch_dim is false, so as_tensor and channel_first do not have effect')
        as_tensor, channel_first = False, False
    if batch_dim:
        input_data = np.expand_dims(input_data, 0)
        if channel_first:
            input_data = np.transpose(input_data, [0, 3, 1, 2])
        if as_tensor:
            input_data = tf.constant(input_data)
    if type == 'ImageNet':
        input_data = (input_data - 0.45) / 0.225
    return input_data, (original_height, original_width)


def visualize(model: DepthEstimationModel = None, channel_first=False):
    outputs = model.disp
    original_height, original_width = model.original_size
    print("H: %d, W: %d"%(original_height, original_width), type(original_height))

    print("-> Visualizing...")
    if channel_first:
        outputs = np.transpose(outputs, [0, 2, 3, 1])
        print("Change to channel-last, now: ", outputs.shape)
    disp_np = np.squeeze(outputs)
    # output_name = os.path.splitext(os.path.basename(image_path))[0]
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)

    colormapped_im = imutils.resize(colormapped_im, width=original_width, height=original_height)

    print(type(colormapped_im), colormapped_im.shape)
    # plt.imshow(colormapped_im), plt.show()

    return colormapped_im


def process_enc_outputs(features_raw, enc_model='pb', dec_model='pb'):
    """convert featrues from encoder to proper form for decoder input
    encoder/decoder model type: 'pb' or 'keras'
    Note:
        - For outputs: keras gives LIST; saved_model (.pb) gives DICTIONARY
        - For inputs: for multiple inputs, keras takes TUPLE; saved_model takes DICTIONARY,or keyword-specific arguments
    Examples:
    1) enc_pb -> decoder_pb: 'dict' -> 'dict'
    2) enc_keras -> decoder_keras 'list' -> 'tuple'
    """
    model_types = ['pb', 'keras']
    if enc_model not in model_types and dec_model not in model_types:
        raise NotImplementedError

    if enc_model == model_types[0]:
        if dec_model == model_types[0]:
            features = {}
            for i in range(1, 6):
                features['input_%d' % i] = (features_raw['output_%d' % i])
        elif dec_model == model_types[1]:
            features = []
            for i in range(1, 6):
                features.append(features_raw['output_%d' % i])
                features = tuple(features)

    elif enc_model == model_types[1]:
        if dec_model == model_types[0]:
            features = {}
            for i in range(1, 6):
                features['input_%d' % i] = features_raw[i]
        elif dec_model == model_types[1]:
            features = tuple(features_raw)

    else:
        raise NotImplementedError

    return features


def init_models(model: DepthEstimationModel):
    # encoder = tf.keras.models.load_model("encoder_res18_singlet")
    enc_imported = tf.saved_model.load(model.encoder_path)
    print("encoder loaded")

    # encoder = tf.keras.models.load_model("encoder_res18_singlet")
    dec_imported = tf.saved_model.load(model.decoder_path)
    print("decoder loaded")
    return enc_imported, dec_imported


def run_estimation(image, model: DepthEstimationModel, show_result:bool= True, data_path:str= None, timer_on=True):
    colormapped_img, scaled_disp = None, None
    model.original_size = image.shape[:2]
    if model.encoder_net is None or model.decoder_net is None:
        model.encoder_net, model.decoder_net = init_models(model)

    timers = defaultdict(lambda:1e-4)
    t1 = time.time()
    encoder = model.encoder_net.signatures['serving_default']
    decoder = model.decoder_net.signatures['serving_default']
    model.clear_results()
    input_data, model.size = prepare_image(image, batch_dim=True, as_tensor=True, channel_first=False)
    timers['preprocess'] = time.time() - t1

    t1 = time.time()
    model.features = encoder(input_data)
    dec_input = process_enc_outputs(model.features, enc_model='pb', dec_model='pb')
    timers['encoding'] = time.time() - t1

    t1 = time.time()
    disp_raw = decoder(**dec_input)
    for k, v in disp_raw.items():
        model.disp = v
    timers['decoding'] = time.time() - t1

    if show_result:
        t1 = time.time()
        colormapped_img = visualize(model)
        timers['visualizing'] = time.time() - t1
    if data_path:
        # todo: which one is the depth?
        scaled_disp, depth = postprocess(model.disp, depth_npy_path=data_path)
        print("depth value saving... in %s " % data_path)
    if timer_on:
        print_runtime(timers, image=None)
    return colormapped_img, scaled_disp


def postprocess(disp, depth_npy_path=None):
    import os

    def disp_to_depth(disp, min_depth, max_depth):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1. / max_depth
        max_disp = 1. / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1. / scaled_disp
        return scaled_disp, depth

    scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

    if depth_npy_path is not None:
        if not os.path.exists(depth_npy_path):
            os.makedirs(depth_npy_path)
        np.save(depth_npy_path, depth.numpy())

    return scaled_disp


def print_runtime(timers, image=None):
    """ all timers stored in a dictonary"""
    all_ms, all_fps = '', ''
    for name, val in timers.items():
        all_ms = ''.join([all_ms, name,': ', '%.1f'%(val*1000), 'ms | '])
        all_fps = ''.join([all_fps, name,': ', '%.1f'%(1./(val+1e-5)), 'FPS | '])
    print(all_ms)
    print(all_fps)
    if image is not None:
        fontScale, fontFace = 0.5,  0
        text_size, base_line = cv.getTextSize(all_ms, fontFace, fontScale, 1)
        cv.putText(image, all_ms, (10, 20), fontFace, fontScale, color=(255,255,255), thickness=1)
        cv.putText(image, all_fps, (10, 40),
                   fontFace, fontScale, color=(255, 255, 255), thickness=1)



