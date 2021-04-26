import os
import tensorflow as tf
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import pickle
import time
from collections import defaultdict
from utils import *
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from new_trainer import build_models
from absl import app, flags
import datetime
from src.trainer_helper import colorize

# tf.keras.backend.set_floatx('float16')
CMAP_DEFAULT = 'plasma'
SCALING = 10  # 10
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'assets/test_image.jpg', 'path to a video or an image file')
flags.DEFINE_string('weights_dir', 'logs/weights/epoch_13/weights_090_79', 'load weights from')
flags.DEFINE_string('save_result_to', 'outputs', 'if set, results will be saved to that path')
flags.DEFINE_bool('save_concat_image', True, 'if set True, result will include original image')
flags.mark_flag_as_required('data_path')
flags.mark_flag_as_required('weights_dir')


def main(_):
    if FLAGS.data_path.endswith(('.mp4', '.avi')):
        detect_video(FLAGS)
    elif FLAGS.data_path.endswith(('.png', '.jpg')):
        detect_image(FLAGS)
    else:
        raise NotImplementedError


def detect_image(opt):
    print('-> Loading models...')
    models = load_models(opt.weights_dir)
    image = cv.imread(opt.data_path)
    orig_h, orig_w = image.shape[:2]

    print('-> Run detection on image...')
    scaled_size = (640, 192)
    input_data, _ = prepare_image(image, as_tensor=True, feed_size=scaled_size, image_type='normal')
    disp_raw = models['depth_dec'](models['depth_enc'](input_data))

    disp = disp_raw['output_0']
    output, colormapped_small = visualize(disp, original_width=orig_w, original_height=orig_h)
    # output = colorize(disp, cmap='plasma')

    if opt.save_concat_image:
        output = np.concatenate([output, image], axis=0)
    if opt.save_result_to != '':
        save_path = os.path.join(opt.save_result_to,
                                 'test_image{}.jpg'.format(current_time))
        print('-> Saveing image to', save_path)
        cv.imwrite(save_path, output)
        # plt.savefig(save_path)
    plt.imshow(output), plt.show()


def detect_video(opt):
    models = load_models(opt.weights_dir)

    cap = cv.VideoCapture(opt.data_path)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    scaled_size = (640, 192)

    if opt.save_result_to != '':
        print('-> Results will be saved to', opt.save_result_to)
        save_size = (orig_w, orig_h * 2) if opt.save_concat_image else (orig_w, orig_h)
        writer = cv.VideoWriter(os.path.join(opt.save_result_to, 'test_video{}.avi'.format(current_time)),
                                fourcc, 12, save_size)

    print('-> Detecting video...')
    print('cap on?', cap.isOpened())
    timers = defaultdict(lambda: 1e-4)
    while cap.isOpened():
        ok, image = cap.read()
        if type(image) is None or not ok:
            exit("wrong video")
        t0 = time.perf_counter() * 1000.

        t1 = time.perf_counter() * 1000.
        input_data, _ = prepare_image(image, as_tensor=True, feed_size=scaled_size, image_type='normal')
        features = models['depth_enc'](input_data)
        timers['encoder'] = time.perf_counter() * 1000. - t1

        t1 = time.perf_counter() * 1000.
        disp_raw = models['depth_dec'](features)
        disp = disp_raw['output_0']
        timers['decoder'] = time.perf_counter() * 1000. - t1

        t1 = time.perf_counter() * 1000.
        output, colormapped_small = visualize(disp, original_width=orig_w, original_height=orig_h)
        timers['visualize'] = time.perf_counter() * 1000. - t1

        # check_depth_within_box(colormapped_small, disp)

        timers['overall'] = time.perf_counter() * 1000. - t0
        print_runtime(timers, image=output)

        if opt.save_concat_image:
            output = np.concatenate([output, image], axis=0)
        cv.imshow('', output)
        if opt.save_result_to != '':
            writer.write(output)
        if cv.waitKey(1) & 0xFF == 27:
            break
    print('-> Done')


def load_models(weights_dir):
    print('-> Loading models...')
    models = {
        'depth_enc': ResNet18_new(),
        'depth_dec': DepthDecoder_full()
    }
    models = build_models(models)
    for k, model in models.items():
        model_path = os.path.join(weights_dir, k+'.h5')
        model.load_weights(model_path)
    return models


def prepare_image(image=None, feed_size=(192, 640),
                  as_tensor=True, image_type='ImageNet'):
    feed_width, feed_height = feed_size
    if image is None:
        image_path = 'assets/test_image.jpg'
        image = cv.imread(image_path)
    original_height, original_width = image.shape[:2]
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    if max(image.shape[:2]) > max(feed_width, feed_height):
        interp_op = cv.INTER_AREA
    else:
        interp_op = cv.INTER_CUBIC
    input_data = cv.resize(image, (feed_width, feed_height), interp_op)

    input_data = input_data.astype(np.float32) / 255.
    if as_tensor:
        input_data = tf.expand_dims(input_data, 0)

    if image_type == 'ImageNet':
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
    colormapped_small = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)
    # im.save("{}_disp.jpeg".format(output_name))
    colormapped_normal = cv.resize(colormapped_small, (original_width, original_height))
    colormapped_normal = cv.cvtColor(colormapped_normal, cv.COLOR_BGR2RGB)
    return colormapped_normal, colormapped_small


def print_runtime(timers, image=None):
    """ all timers stored in a dictonary"""
    all_ms, all_fps = '', ''
    for name, val in timers.items():
        if val < 1:
            continue
        all_ms = ''.join([all_ms, name, ': ', '%.1f' % (val), 'ms | '])
        all_fps = ''.join([all_fps, name, ': ', '%.1f' % (1000. / (val)), 'FPS | '])
    print(all_ms)
    print(all_fps)
    if image is not None:
        fontScale, fontFace = 0.5, 0
        text_size, base_line = cv.getTextSize(all_ms, fontFace, fontScale, 1)
        cv.putText(image, all_ms, (10, 20), fontFace, fontScale, color=(255, 255, 255), thickness=1)
        cv.putText(image, all_fps, (10, 40),
                   fontFace, fontScale, color=(255, 255, 255), thickness=1)


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


""" ----- Below is Not Important for simple run """


def draw_box(image, bboxes, img_h=416, img_w=128):
    bbox_thick = int(0.5 * (img_h + img_w) / 500)
    if type(bboxes[0]) == int or type(bboxes[0]) == float:
        bboxes = [bboxes]  # in case the input is a single bbox (by mistake use)
    for bbox in bboxes:
        l, t, w, h = bbox
        c1, c2 = (int(l), int(t)), (int(w + l), int(h + t))
        cv.rectangle(image, c1, c2, (100, 200, 100), bbox_thick)


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

    scaling = (1, 1)
    image = cv.resize(image, scaled_size, cv.INTER_AREA)
    print("scaling", scaling)
    # plt.imshow(depth_map), plt.show()
    for bbox in bboxes:
        xl, yl, w, h = bbox[0] * scaling[0], bbox[1] * scaling[1], bbox[2] * scaling[0], bbox[3] * scaling[1]
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


def check_depth_within_box(colormapped_small, disp):
    print("checking depth... ")
    disp = np.squeeze(disp)
    # depth = 1./disp
    _, depth = disp_to_depth(disp, 0.1, 100)
    roi = cv.selectROI('', colormapped_small)
    print("roi", roi)
    xl, yl = roi[0], roi[1]
    xr, yr = xl + roi[2], yl + roi[3]
    print("box in depthmap: [%d:%d, %d:%d]" % (yl, yr, xl, xr))
    draw_box(colormapped_small, roi)

    # print("mean: ", scaled_disp_mean, " : ", depth_mean)
    tgt_disp = disp[yl:yr, xl:xr]
    tgt_depth = depth[yl:yr, xl:xr]

    scaled_disp_mean, tgt_disp_valid = get_median_range(tgt_disp)
    depth_mean, depth_valid = get_median_range(tgt_depth)

    print("mean: ", scaled_disp_mean, depth_mean)

    fig = plt.figure(figsize=(2, 1))
    fig.add_subplot(2, 1, 1)
    plt.plot(list(range(tgt_disp.size)), np.sort(tgt_disp.flatten()))
    plt.plot(list(range(tgt_disp_valid[0][0], tgt_disp_valid[0][1])), tgt_disp_valid[1])
    fig.add_subplot(2, 1, 2)
    plt.plot(list(range(tgt_depth.size)), np.sort(tgt_depth.flatten()))
    plt.plot(list(range(depth_valid[0][0], depth_valid[0][1])), depth_valid[1])
    plt.show()


if __name__ == '__main__':
    app.run(main)
