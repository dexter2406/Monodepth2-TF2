import os, shutil
import pickle
import numpy as np
import tensorflow as tf

from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder
from src.trainer_helper import build_models

rootdir = os.path.dirname(__file__)


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


def make_hom_intrinsics(intrinsic_mat, same_video):
    """ Make homogenous intrinsics
    As input of
    Args:
        intrinsic_mat: Tensor, [B, 2, 3]
            will be padded to be homogenous mat
        same_video: if True, intrinsics_mat will be averaged as identical ones though batch dimension
    Returns:
        hom_intrinsics: Tensor, [B, 4, 4]
    """
    batch_size = intrinsic_mat.shape[0]
    if same_video:
        intrinsic_mat = tf.reduce_mean(intrinsic_mat, axis=0, keepdims=True)
        hom_intrinsics = hom_intrinsics_helper(intrinsic_mat, 1)
        hom_intrinsics = tf.concat([hom_intrinsics] * batch_size, axis=0)
    else:
        batch_size = intrinsic_mat.shape[0]
        hom_intrinsics = hom_intrinsics_helper(intrinsic_mat, batch_size)

    return hom_intrinsics


def get_models(weights_dir, exp=True):
    models = {
        'depth_enc': ResNet18_new(norm_inp=True),
        'depth_dec': DepthDecoder_full(),
        'pose_enc': ResNet18_new(norm_inp=True),
        'pose_dec': PoseDecoder(num_frames_to_predict_for=1 if exp else 2)
    }
    build_models(models, rgb_cat_depth=True if exp else False)

    if weights_dir == '':
        weights_dir = 'logs/weights/pretrained_resnet18'

    for m_name, model in models.items():
        weights_name = m_name
        if exp:
            if m_name == 'pose_dec':
                weights_name = m_name + '_one_out'
            if m_name == 'pose_enc':
                weights_name = m_name + '_concat'
        path = os.path.join(weights_dir, weights_name+'.h5')
        if not os.path.isfile(path):
            print('%s not found, skipping' % path)
            continue
        models[m_name].load_weights(path)
    return models


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
    all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec', 'intrinsics_head']

    print("-> Check options...")
    for m in FLAGS.models_to_load:
        if m not in all_models:
            raise ValueError("\t'%s' is not supported model, choose from: " % m, all_models)
    print('\t Using dataset:', FLAGS.split)

    if not FLAGS.learn_intrinsics:
        FLAGS.use_min_proj = False

    if FLAGS.add_pose_loss:
        print('calc_reverse_transform automatically set `True`, because `add_pose_loss` requires it')
        FLAGS.include_revers = True

    if (FLAGS.padding_mode == 'zeros' and not FLAGS.mask_border) or \
        (FLAGS.padding_mode == 'border' and FLAGS.mask_border):
        print('either `zero_padding - masking`, or `border_padding - no_masking` ')

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
            else:
                print('\t No weights_dir specified: decoders from scratch, encoders with pretrained weights')
        else:
            not_loaded = [m for m in all_models if m not in FLAGS.models_to_load]

            if FLAGS.from_scratch:
                FLAGS.models_to_load = []
                print('\ttraining from scratch, NO WEIGHTS LOADED even if --models_to_load specifies them.')
            else:
                if len(not_loaded) != 0:
                    raise ValueError('if not from scratch, please use default setting for '
                                     '--models_to_load to load all models')
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

