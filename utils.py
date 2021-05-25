import os, shutil
import pickle
import numpy as np
import tensorflow as tf

rootdir = os.path.dirname(__file__)


def dilate_box(box):
    """dilate one single box to ensure coverage
    """
    offset_ratio = [-1, 0.5]     # up/down. left/right direction
    dilate_factor = [0.4, -0.2]     # in H, W direction, respectively
    box = tf.cast(box, tf.float32)

    h_new = (box[3] - box[1]) * dilate_factor[0]
    box_t = box[1] - h_new * offset_ratio[0]
    box_b = box[3] + h_new * (1-offset_ratio[0])

    w_new = (box[2] - box[0]) * dilate_factor[1]
    box_l = box[0] - w_new * offset_ratio[1]
    box_r = box[2] + w_new * (1-offset_ratio[1])
    out = list(map(int, [box_l, box_t, box_r, box_b]))
    return out

def merge_boxes(boxes1, boxes2):
    """merge boxes1 and boxes2
    Args:
        boxes1: Tensor, shape (B, 4) or (4, ), box batch
        boxes2: same as boxes1
    Returns:
        a box with merged area
    """
    def make_batch_dim(x):
        if isinstance(x, (tuple, list)):
            x = tf.constant(x)
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        return x
    boxes1 = make_batch_dim(boxes1)
    boxes2 = make_batch_dim(boxes2)

    boxes = tf.stack([boxes1, boxes2], axis=-2)  # (.,2,4)
    box_min = tf.transpose(
        tf.reduce_min(tf.transpose(boxes), axis=-2, keepdims=True)[:2])  # (.,1,4)
    box_max = tf.transpose(
        tf.reduce_max(tf.transpose(boxes), axis=-2, keepdims=True)[2:])
    box_merged = tf.concat([box_min, box_max], axis=-1)  # (.,4)
    return tf.squeeze(box_merged)


def crop_to_aspect_ratio(image, feed_size):
    h, w = image.shape[:2]
    asp_ratio = feed_size[1] / feed_size[0]   # W/H, usually 640/192
    tolerance = 0.05
    if abs(w / h - asp_ratio) > tolerance:
        h_goal = int(w // asp_ratio)
        h_start = int((h - h_goal) * 0.5)
        image = image[h_start: h_start + h_goal, :w, :]
    return image


def assert_valid_hom_intrinsics(intrinsics_mat):
    shapes = list(intrinsics_mat.shape)
    if len(shapes) != 3 or shapes[1] != shapes[2] or shapes[2] != 4:
        raise ValueError('intrinsics shape should be [B,4,4], got {}'.format(shapes))


def hom_intrinsics_helper(intrinsics_mat, batch_num):
    """Pad a [B,2,3] or [2,3] intrinsics to be homogenous"""
    K_shape = intrinsics_mat.shape

    if K_shape[-2] != 2 or K_shape[-1] not in [3, 4]:
        raise ValueError('intrinsics shape is wrong: {}'.format(K_shape))

    if len(K_shape) == 2:
        # in case no batch dim
        intrinsics_mat = tf.expand_dims(intrinsics_mat, axis=0)

    if K_shape[0] == 1 and K_shape[0] != batch_num:
        # in case batch_dim=1 but needs more
        intrinsics_mat = tf.concat([intrinsics_mat]*batch_num, axis=0)

    if K_shape[-1] == 4:
        # if it's half-way padded [B,2,4], just restore to [B,2,3]
        intrinsics_mat = intrinsics_mat[..., :3]

    # now K should have shape [B, 3, 3]
    row_3 = tf.cast(tf.tile([[[0, 0, 1]]], [batch_num, 1, 1]), dtype=tf.float32)
    row_4 = tf.cast(tf.tile([[[0, 0, 0]]], [batch_num, 1, 1]), dtype=tf.float32)
    last_col = tf.cast(tf.reshape(tf.tile([[[0, 0, 0, 1]]], [batch_num, 1, 1]), [batch_num, 4, 1]),
                       dtype=tf.float32)
    hom_intrinsics = tf.concat([intrinsics_mat, row_3, row_4], axis=1)
    hom_intrinsics = tf.cast(tf.concat([hom_intrinsics, last_col], axis=2), dtype=tf.float32)
    return hom_intrinsics


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


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
                print('input_%d : ' % i, features_raw[i-1].shape)
                features['input_%d' % i] = features_raw[i-1]
        elif dec_model == model_types[1]:
            features = tuple(features_raw)

    else:
        raise NotImplementedError

    return features


def get_median_range(mtx):
    max_val = 30
    SCALING = 10
    """distance matrix for one patch"""
    ranging = [.4, .6]
    depth_all = np.sort(mtx.flatten())
    val_idx = [int(ranging[0] * depth_all.size), int(ranging[1] * depth_all.size)]
    depth_valid = depth_all[val_idx[0]: val_idx[1]]
    # print(depth_valid)
    # todo: disp converted to real depth values, maybe see Struct2Depth?
    # depth_valid = max_val - np.mean(depth_valid) * SCALING
    # depth_valid_mean = np.mean(depth_valid)
    print("depth_all shape", depth_all.shape)
    print("depth_valid shape", depth_valid.shape)
    depth_valid_mean = sum(depth_valid) / depth_valid.size
    print("random value:", depth_valid[1000])
    return depth_valid_mean, [val_idx, depth_valid]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def del_files(root_dir):
    for f in os.listdir(root_dir):
        file_path = os.path.join(root_dir, f)
        try:
            shutil.rmtree(file_path)
        except OSError:
            os.remove(file_path)


def check_options(FLAGS):
    all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec']

    print("-> Check options...")
    for m in FLAGS.models_to_load:
        if m not in all_models:
            raise ValueError("\t'%s' is not supported model, choose from: " % m, all_models)
    print('\t Using dataset:', FLAGS.split)
    if FLAGS.debug_mode:
        print('\t[debug mode] Check intermediate results. '
              'Models will not be trained or saved even if options are given.')
        FLAGS.batch_size = 2
        print("\t[debug mode] batch_size is reset to %d\n" % FLAGS.batch_size)

    else:
        print("\tTURN ON @tf.function for grad() and DataProcessor!")
        save_dir = FLAGS.save_model_path
        if FLAGS.save_model_path == '':
            if FLAGS.save_root != '':
                rootdir = FLAGS.save_root
            else:
                rootdir = os.path.dirname(__file__)
            save_dir = os.path.join(rootdir, 'logs/weights')
            print('\tno save_model_path specified, use %s instead' % FLAGS.save_model_path)

        save_path = os.path.join(save_dir, FLAGS.model_name)
        if not os.path.isdir(save_path) and not FLAGS.debug_mode:
            os.makedirs(save_path)
        FLAGS.save_model_path = save_path
        print('\tweights will be saved in folder {}'.format(save_path))

        if FLAGS.train_depth: print("\ttraining depth encoder-decoder")
        if FLAGS.train_pose: print("\ttraining pose encoder-decoder")

        from_scratch, weights_dir = FLAGS.from_scratch, FLAGS.weights_dir
        if weights_dir == '':
            if from_scratch:
                print("\n\tall models are trained from scratch")
                if FLAGS.opt.add_rot_loss:
                    raise ValueError('rot_error should NOT be activated when from scratch')
            else:
                print('\t No weights_dir specified: decoders from scratch, encoders with pretrained weights')
        else:
            not_loaded = [m for m in all_models if m not in FLAGS.models_to_load]

            if FLAGS.from_scratch:
                FLAGS.models_to_load = []
                print('\ttraining from scratch, NO WEIGHTS LOADED even if --models_to_load specifies them.')
            else:
                if len(not_loaded) != 0:
                    print('not loaded:', not_loaded)
                else:
                    print("\twill be loaded %s, %s will be randomly initialized" % (FLAGS.models_to_load, not_loaded))

        if FLAGS.recording:
            print('\tSummary will be recorded')
        else:
            print('\tSummary will not be recorded')

    print("\tbatch size: %d, epoch number: %d" % (FLAGS.batch_size, FLAGS.num_epochs))


def save_features(features):
    """
    features: List of torch.Tensor
        will be converted to List of numpy.ndarray, and saved
    """
    output = [to_numpy(f) for f in features]
    with open('raw_features.pkl', 'wb') as df:
        pickle.dump(output, df)


def get_depth_savedmodel(get_enc=True, get_dec=True, enc_keras=False, dec_keras=False):
    """get SavedModel
    usage example:
    enc_imported, dec_imported = get_models()
    encoder = enc_imported.signatures['serving_default']
    encoder = dec_imported.signatures['serving_default']

    features = process_enc_outputs(feature_raw, enc_model='pb', dec_model='pb')
    # for multiple-inputs SavedModel, inputs need to be wrapped in a dict with correct input_tensor names
    disp_raw = models['depth_dec'](**features)
    """
    enc_imported, dec_imported = None, None

    if get_dec:
        dec_name = "models/depth_decoder_fullout"  # depth_decoder_oneout
        if not dec_keras:
            # decoder = tf.keras.models.load_model(dec_name)
            dec_imported = tf.saved_model.load(dec_name)
            # decoder = dec_imported.signatures['serving_default']
            print("decoder SavedModel loaded")
        else:
            dec_imported = tf.keras.models.load_model(dec_name)
            print("decoder Keras model loaded")

    if get_enc:
        enc_name = "models/encoder_res18_singlet"
        if not enc_keras:
            # encoder = tf.keras.models.load_model("encoder_res18_singlet")
            enc_imported = tf.saved_model.load(enc_name)
            # encoder = enc_imported.signatures['serving_default']
            print("encoder SavedModel loaded")
        else:
            enc_imported = tf.keras.models.load_model(enc_name)
            print("encoder Keras model loaded")

    return enc_imported, dec_imported

