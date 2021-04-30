from absl import flags
import os
import datetime


rootdir = os.path.dirname(__file__)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec']
FLAGS = flags.FLAGS

# Experimental
flags.DEFINE_bool('exp_mode', True, 'experiment mode')
flags.DEFINE_bool('concat_depth_pred', True, 'concat depth_pred to rgb images for pose net input')
flags.DEFINE_bool('add_pose_loss', True, 'add pose loss to training')
# todo: hyperparameters
flags.DEFINE_float('pose_loss_weight', 1, 'weight for pose_loss')
flags.DEFINE_float('smoothness_ratio', 1e-3, 'ratio to calculate smoothness loss')
flags.DEFINE_float('ssim_ratio', 0.85, 'ratio to calculate SSIM loss')
flags.DEFINE_bool('use_depth_consistency', False, 'add depth_consistency to handle occlusion between two frames')
flags.DEFINE_bool('mask_border', False, 'mask out the region padded by bilinear sampler '
                                        'when computing losses (only for zero-padding)')
flags.DEFINE_bool('calc_revserse_transform', True, 'calcualting transformation in reversed temp order, this'
                                                  'must be true when `add_pose_loss` is activated')

# Pre-settings
flags.DEFINE_integer('pose_num', 1, 'number of poses produced by pose decoder')
flags.DEFINE_bool('norm_input', True, 'normalize input -> (-1, 1) for encoders')
flags.DEFINE_string('weights_dir', '',  'the folder that stores weights files.')
flags.DEFINE_list('models_to_load', all_models,
                  'load weights for specified models, by default all of them')
flags.DEFINE_string('model_name', current_time, 'specify a dirname to collect weights, if not, current time is used')
flags.DEFINE_bool('train_depth', True, 'whether to train depth decoder-encoder')
flags.DEFINE_bool('train_pose', True, 'whether to train pose decoder-encoder')
flags.DEFINE_string('save_model_path', '', 'path where weights are saved')
flags.DEFINE_string('data_path', r'F:\Dataset\kitti_raw', 'path that stores corresponding dataset')
flags.DEFINE_string('dataset', 'kitti_raw', 'specify a dataset, choices from [\'kitti_raw\', \'kitti_odom\']')
flags.DEFINE_string('padding_mode', 'border', 'padding mode for bilinear sampler')

# Training
flags.DEFINE_bool('from_scratch', False, 'whether trained from scratch, coorperate with load_weights_folder')
flags.DEFINE_string('run_mode', 'train', 'choose from [\'train\', \'eval_depth\', \'eval_pose\']')
flags.DEFINE_string('split', 'eigen_zhou', 'training split, choose from: '
                                           '["eigen_zhou", "eigen_full", "odom", "benchmark"]')
flags.DEFINE_bool('recording', True, 'whether to write results by tf.summary')
flags.DEFINE_string('record_summary_path', 'logs/gradient_tape/', 'root path to write summary')
flags.DEFINE_integer('record_freq', 250, 'frequency to record')
flags.DEFINE_integer('num_epochs', 10, 'total number of training epochs')
flags.DEFINE_integer('batch_size', 6, 'batch size')
flags.DEFINE_bool('debug_mode', False, 'inspect intermediate results')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate')
flags.DEFINE_integer('lr_step_size', 5, 'step size to adapt learning rate (piecewise)')
flags.DEFINE_integer('val_num_per_epoch', 10, 'validate how many times per epoch')

# Model-related
flags.DEFINE_integer('num_scales', 4, 'number of scales')
flags.DEFINE_integer('src_scale', 0, 'source scale')
flags.DEFINE_integer('height', 192, 'height of input image')
flags.DEFINE_integer('width', 640, 'width of input image')
flags.DEFINE_list('frame_idx', [0, -1, 1], 'index of target, previous and next frame')
flags.DEFINE_bool('do_augmentation', True, 'apply image augmentation')
flags.DEFINE_bool('do_automasking', True, 'apply auto masking')
flags.DEFINE_float('min_depth', 0.1, 'minimum depth when applying scaling/normalizing to depth estimates')
flags.DEFINE_float('max_depth', 100., 'maximum depth when applying scaling/normalizing to depth estimates')

# Evaluation
flags.DEFINE_bool('eval_eigen_to_benchmark', False, '?')
flags.DEFINE_bool('save_pred_disps', False, 'save generated dispairty maps')
flags.DEFINE_bool('no_eval', False, 'don\'t conduct evaluation for debugging or saving preds')
flags.DEFINE_float('pred_depth_scale_factor', 1., 'additional depth scaling factor')
flags.DEFINE_bool('use_median_scaling', True, 'use median filter to calculate scaling ratio')
flags.DEFINE_string('eval_split', 'eigen', 'evaluation split, choose from: '
                                           '["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"]')
flags.DEFINE_string('use_ext_res', None, 'use imported disparity predictions for evaluation, '
                                         'instead of generating them now')
flags.mark_flag_as_required('run_mode')


def get_options():
    return FLAGS
