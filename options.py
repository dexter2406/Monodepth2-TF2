from absl import flags
import os
import datetime


rootdir = os.path.dirname(__file__)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec']
FLAGS = flags.FLAGS


flags.DEFINE_boolean('use_sampler_mask',False,  'if set, use zero-padding and mask out '
                                                'invalid region for bilinear sampler')
flags.DEFINE_bool('add_rot_loss',       False,  'if set, use rotation consistency loss')
flags.DEFINE_float('rot_loss_w',        1e-3,   'weight of rotation loss')
flags.DEFINE_float('reproj_loss_w', 1., 'weight for reprojection loss')
flags.DEFINE_float('smoothness_ratio',  1e-3, 'ratio to calculate smoothness loss')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate')
flags.DEFINE_boolean('depth_norm_inp',  True,   'if set, input for depth_enc will be normalized')
flags.DEFINE_bool('train_depth',        True,   'whether to train depth decoder-encoder')
flags.DEFINE_bool('train_pose',         True,   'whether to train pose decoder-encoder')

# Pre-settings
flags.DEFINE_string('weights_dir',      '',     'the folder that stores weights files.')
flags.DEFINE_string('save_model_path',  '',     'path where weights are saved')
flags.DEFINE_string('save_root',        '',     'another root path to store logs')
flags.DEFINE_list('models_to_load',     all_models, 'load weights for specified models, by default all of them')
flags.DEFINE_string('data_path',        r'F:\Dataset\kitti_raw', 'path that stores corresponding dataset')
flags.DEFINE_string('dataset',          'kitti_raw', 'specify a dataset, choices from [\'kitti_raw\', \'kitti_odom\']')
flags.DEFINE_bool('disable_gt',         False, 'disable ground truth')
flags.DEFINE_string('model_name',       current_time, 'specify a dirname to collect weights')
flags.DEFINE_bool('from_scratch',       False,  'whether trained from scratch, coorperate with load_weights_folder')


# Training
flags.DEFINE_string('run_mode', 'train', 'choose from [\'train\', \'eval_depth\', \'eval_pose\']')
flags.DEFINE_string('split', 'eigen_zhou', 'training split, choose from: '
                                           '["eigen_zhou", "eigen_full", "odom", "benchmark"]')
flags.DEFINE_bool('recording', True, 'whether to write results by tf.summary')
flags.DEFINE_string('record_summary_path', 'logs/gradient_tape/', 'root path to write summary')
flags.DEFINE_integer('record_freq', 250, 'frequency to record')
flags.DEFINE_integer('num_epochs',  10, 'total number of training epochs')
flags.DEFINE_integer('batch_size',  6, 'batch size')
flags.DEFINE_bool('debug_mode',     False, 'inspect intermediate results')
flags.DEFINE_integer('lr_step_size', 5, 'step size to adapt learning rate (piecewise)')
flags.DEFINE_integer('val_num_per_epoch', 10, 'validate how many times during each epoch')
# Model-related
flags.DEFINE_integer('num_scales',  4, 'number of scales')
flags.DEFINE_integer('src_scale',   0, 'source scale')
flags.DEFINE_integer('height',      192, 'height of input image')
flags.DEFINE_integer('width',       640, 'width of input image')
flags.DEFINE_list('frame_idx',      [0, -1, 1], 'index of target, previous and next frame')
flags.DEFINE_bool('do_augmentation',    True, 'apply image augmentation')
flags.DEFINE_bool('do_automasking',     True, 'apply auto masking')
flags.DEFINE_float('ssim_ratio',    0.85, 'ratio to calculate SSIM loss')
flags.DEFINE_float('min_depth',     0.1, 'minimum depth when applying scaling/normalizing to depth estimates')
flags.DEFINE_float('max_depth',     100., 'maximum depth when applying scaling/normalizing to depth estimates')

# Evaluation
flags.DEFINE_bool('eval_eigen_to_benchmark', False, '?')
flags.DEFINE_bool('save_pred_disps',        False, 'save generated dispairty maps')
flags.DEFINE_bool('no_eval',                False, 'don\'t conduct evaluation for debugging or saving preds')
flags.DEFINE_float('pred_depth_scale_factor', 1., 'additional depth scaling factor')
flags.DEFINE_bool('use_median_scaling',     True, 'use median filter to calculate scaling ratio')
flags.DEFINE_string('eval_split', 'eigen', 'evaluation split, choose from: '
                                           '["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"]')
flags.DEFINE_string('use_ext_res', None, 'use imported disparity predictions for evaluation, '
                                         'instead of generating them now')

def get_options():
    return FLAGS
