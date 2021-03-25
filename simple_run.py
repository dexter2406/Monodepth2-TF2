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

# tf.keras.backend.set_floatx('float16')

def get_models():
    dec_name = "models/depth_decoder_singlet" # depth_decoder_singlet
    # decoder = tf.keras.models.load_model(dec_name)
    dec_imported = tf.saved_model.load(dec_name)
    decoder = dec_imported.signatures['serving_default']
    print("decoder loaded")

    # encoder = tf.keras.models.load_model("encoder_res18_singlet")
    enc_imported = tf.saved_model.load("models/encoder_res18_singlet")
    encoder = enc_imported.signatures['serving_default']
    print("encoder loaded")
    return enc_imported, dec_imported


def test():
    enc_imported, dec_imported = get_models()
    encoder = enc_imported.signatures['serving_default']
    decoder = dec_imported.signatures['serving_default']
    # return encoder, decoder
    vid_path = [r"D:\MA\motion_ds\SeattleStreet_1.mp4",
                r"D:\MA\Struct2Depth\KITTI_odom_02\seq2.avi"]
    cap = cv.VideoCapture(vid_path[1])
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    scaled_size = (640,192)
    writer = cv.VideoWriter('depth_monodepth2.avi', fourcc, 12, (1240, 372))
    df_b = open("bboxes.txt", 'rb')

    timers = defaultdict(lambda : 1e-4)
    while cap.isOpened():
        ret, image = cap.read()
        if type(image) is None:
            exit("wrong video")
        t0 = time.time()
        # t1 = time.time()
        input_data, (_, _) = prepare_image(image, batch_dim=True, as_tensor=True, channel_first=False)
        # timers['prepare'] = time.time() - t1

        t1 = time.time()
        feature_raw = encoder(input_data)
        timers['encoder'] = time.time() - t1

        t1 = time.time()
        features = process_enc_outputs(feature_raw, enc_model='pb', dec_model='pb')
        timers['feature process'] = time.time() - t1

        t1 = time.time()
        disp_raw = decoder(**features)
        for k, v in disp_raw.items():
            disp = v
        timers['decoder'] = time.time() - t1

        # scaled_disp = postprocess(disp) # get depth values

        t1 = time.time()
        colormapped_im = visualize(disp, original_width=orig_w, original_height=orig_h)
        timers['visualize'] = time.time() - t1
        timers['overall'] = time.time() - t0
        print_runtime(timers, image=None)
        cv.imshow('', colormapped_im)
        # writer.write(colormapped_im)
        if cv.waitKey(1) & 0xFF==27:
            break


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


def visualize(outputs, original_width, original_height, channel_first=False):
    print("-> Visualizing...")
    if channel_first:
        outputs = np.transpose(outputs[3], [0, 2, 3, 1])
        print("Change to channel-last, now: ", outputs.shape)
    disp_np = np.squeeze(outputs)
    # output_name = os.path.splitext(os.path.basename(image_path))[0]
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # im.save("{}_disp.jpeg".format(output_name))
    colormapped_im = imutils.resize(colormapped_im, width=original_width, height=original_height)

    return colormapped_im


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


def draw_box(image, bboxes, img_h=416, img_w=128):
    bbox_thick = int(0.5 * (img_h + img_w) / 500)
    for bbox in bboxes:
        l, t, w, h = bbox
        c1, c2 = (int(l), int(t)), (int(w+l), int(h+t))
        cv.rectangle(image, c1, c2, (100,200,100), bbox_thick)


def calc_depth_within_bbox(mtx, bboxes, image, scaled_size, show_depth=True):
    def get_median_range(mtx):
        SCALING = 10
        """distance matrix for one patch"""
        ranging = [.4, .6]
        depth_all = np.sort(mtx.numpy().flatten())
        val_idx = [int(ranging[0] * len(depth_all)), int(ranging[1] * len(depth_all))]
        depth_valid = depth_all[val_idx[0]:val_idx[1]]
        # print(depth_valid)
        return np.mean(depth_valid) * SCALING

    scaling=(1,1)
    image = cv.resize(image, scaled_size, cv.INTER_AREA)
    print("scaling", scaling)
    # plt.imshow(depth_map), plt.show()
    for bbox in bboxes:
        xl, yl, w, h = bbox[0]*scaling[0], bbox[1]*scaling[1], bbox[2]*scaling[0], bbox[3]*scaling[1]
        xl, yl, xr, yr = int(xl), int(yl), int(xl + w), int(yl + h)
        mtx_patch = mtx[yl:yr, xl:xr]
        mean_depth = get_median_range(mtx_patch)

        if show_depth:
            msg = '%.2f' % mean_depth
            bbox_thick = int(0.5 * np.sum(mtx.shape) / 500)
            t_size = cv.getTextSize(msg, 0, fontScale=.5, thickness=bbox_thick // 2)[0]
            c3 = (xl + t_size[0], yl - t_size[1] - 3)
            cv.rectangle(image, (xl, yl), (np.float32(c3[0]), np.float32(c3[1])), (50, 200, 50), -1)  # filled
            cv.putText(image, msg, (xl, np.float32(yl - 2)), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), bbox_thick // 2, lineType=cv.LINE_AA)
    return image

if __name__ == '__main__':
    test()
