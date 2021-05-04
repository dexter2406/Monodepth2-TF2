from absl import flags
import os
import datetime


rootdir = os.path.dirname(__file__)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
models_to_load = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec']

# Experimental
flags.DEFINE_boolean('add_mask_loss',   True, 'regularize mask from shrinking to zero')
flags.DEFINE_bool('train_depth',        False, 'whether to train depth decoder-encoder')
flags.DEFINE_bool('train_pose',         False, 'whether to train pose decoder-encoder')
flags.DEFINE_bool('do_automasking',     True, 'apply auto masking')
flags.DEFINE_boolean('exp_mode',        True, 'experiment mode')
flags.DEFINE_boolean('concat_depth',    True, 'concat depth_pred to rgb images for pose net input')
flags.DEFINE_boolean('use_cycle_consistency',   True, 'add depth_consistency to handle occlusion between two frames')
flags.DEFINE_string('padding_mode',     'zeros', 'padding mode for bilinear sampler')
flags.DEFINE_boolean('mask_border',     True,  'mask out the region padded by bilinear sampler '
                                               'when computing losses (only for zero-padding)')
flags.DEFINE_boolean('add_pose_loss',   True, 'add pose loss to training')
flags.DEFINE_boolean('include_revers',  True, 'calculate transformation in reversed temp order, this'
                                              'must be true when `add_pose_loss` is activated')
# NIU: additional depth doesn't seem to help to improve
flags.DEFINE_boolean('use_RGBD',        False, 'use RGB-D instead RGB in reprojection error calculation')
flags.DEFINE_boolean('use_min_proj',    True, 'use minimal projection loss, not suitable for intrinsics training')
flags.DEFINE_boolean('disable_gt',      False, 'disable ground-truth depth')
flags.DEFINE_boolean('learn_intrinsics',False, 'learn intrinsics matrix')
flags.DEFINE_boolean('use_res_trans_loss',      False, 'residual translation error')

# todo: Hyper-parameters
flags.DEFINE_integer('batch_size', 4, 'batch size')
flags.DEFINE_float('smoothness_ratio', 1e-3, 'ratio to calculate smoothness loss')
flags.DEFINE_float('ssim_ratio', 0.85, 'ratio to calculate SSIM loss')
flags.DEFINE_float('reproj_loss_weight', 1., 'reprojection loss weight')
flags.DEFINE_float('cycle_loss_weight', 1., 'weight for cycle-consistency loss')
flags.DEFINE_float('pose_loss_weight', 1e-2, 'weight for pose_loss')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate')
flags.DEFINE_float('mask_loss_w', 5., 'weight for sampler_mask to prevent from shrinking to 0')
flags.DEFINE_float('mask_cover_min', 0.85, 'when using zero-padding, zero-region is usually below certain level')

# Pre-settings
flags.DEFINE_integer('pose_num', 1, 'number of poses produced by pose decoder')
flags.DEFINE_bool('norm_input', True, 'normalize input -> (-1, 1) for encoders')
flags.DEFINE_string('weights_dir', '',  'the folder that stores weights files.')
flags.DEFINE_list('models_to_load', models_to_load,
                  'load weights for specified models, by default all of them')
flags.DEFINE_string('model_name', current_time, 'specify a dirname to collect weights, if not, current time is used')

flags.DEFINE_string('save_model_path', '', 'path where weights are saved')
flags.DEFINE_string('data_path', r'F:\Dataset\kitti_raw', 'path that stores corresponding dataset')
flags.DEFINE_string('dataset', 'kitti_raw', 'specify a dataset, choices from [\'kitti_raw\', \'kitti_odom\']')
flags.DEFINE_string('save_root', '', 'another root path to store logs')
# Training
flags.DEFINE_bool('from_scratch', False, 'whether trained from scratch, coorperate with load_weights_folder')
flags.DEFINE_string('run_mode', 'train', 'choose from [\'train\', \'eval_depth\', \'eval_pose\']')
flags.DEFINE_string('split', 'eigen_zhou', 'training split, choose from: '
                                           '["eigen_zhou", "eigen_full", "odom", "benchmark"]')
flags.DEFINE_bool('recording', True, 'whether to write results by tf.summary')
flags.DEFINE_string('record_summary_path', 'logs/gradient_tape/', 'root path to write summary')
flags.DEFINE_integer('record_freq', 250, 'frequency to record')
flags.DEFINE_integer('num_epochs', 10, 'total number of training epochs')
flags.DEFINE_bool('debug_mode', False, 'inspect intermediate results')
flags.DEFINE_integer('lr_step_size', 5, 'step size to adapt learning rate (piecewise)')
flags.DEFINE_integer('val_num_per_epoch', 10, 'validate how many times per epoch')

# Model-related
flags.DEFINE_integer('num_scales', 4, 'number of scales')
flags.DEFINE_integer('src_scale', 0, 'source scale')
flags.DEFINE_integer('height', 192, 'height of input image')
flags.DEFINE_integer('width', 640, 'width of input image')
flags.DEFINE_list('frame_idx', [0, -1, 1], 'index of target, previous and next frame')
flags.DEFINE_bool('do_augmentation', True, 'apply image augmentation')
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
flags.DEFINE_bool('use_ext_res', False, 'use imported disparity predictions for evaluation'
                                        'instead of generating them now')
flags.DEFINE_string('ext_res_path', '', 'if use_ext_res==True, specify the path to load external result for evaluation')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('run_mode')
flags.mark_flag_as_required('exp_mode')
flags.mark_flag_as_required('use_min_proj')

def get_options():
    return FLAGS
