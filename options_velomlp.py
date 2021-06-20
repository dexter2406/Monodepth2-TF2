from absl import flags
import os
import datetime


rootdir = os.path.dirname(__file__)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_dec']
FLAGS = flags.FLAGS

# For other datasets instead of kitti, settings below might need to be changed:
# --feed_size --split velocity_challenge --dataset velocity --data_path F:/Dataset/VelocityChallenge
flags.DEFINE_float('learning_rate',     1e-3,   'initial learning rate')
flags.DEFINE_bool('debug_mode',         False, 'inspect intermediate results')
flags.DEFINE_integer('frame_interval', 2, 'frame interval for velocity calculation')
flags.DEFINE_float('dist_loss_w', 0.1, "weight for distance loss")
flags.DEFINE_float('velo_loss_w', 1, "weight for distance loss")

# Pre-settings
flags.DEFINE_string('depth_estimator_path', r'D:\MA\motion_ds_experiment\model_data',
                    'the folder that stores weights files for depth encoder-decoder.')
flags.DEFINE_string('save_model_path',  'models/velocity_mlp',     'path where weights are saved')
flags.DEFINE_string('save_root',        '',     'another root path to store logs')
flags.DEFINE_string('data_path',        None, 'path that stores corresponding dataset')
flags.DEFINE_string('model_name',       current_time, 'specify a dirname to collect weights')
flags.DEFINE_bool('from_scratch',       False,  'whether trained from scratch, coorperate with load_weights_folder')
flags.DEFINE_bool('show_image_debug',   False,  'whether show image when debugging')
flags.DEFINE_string('bbox_folder', r'F:\Dataset\VelocityChallenge\benchmark_velocity_train\gt_bboxes'.replace('\\', '/'),
                    'folder to store pre-computed bboxes')

# Training
flags.DEFINE_string('split', 'eigen_zhou', 'training split, choose from: '
                                           '["eigen_zhou", "eigen_full", "odom", "benchmark"]')
flags.DEFINE_bool('recording',  True, 'whether to write results by tf.summary')
flags.DEFINE_string('record_summary_path', 'logs/gradient_tape/', 'root path to write summary')
flags.DEFINE_integer('record_freq', 1000, 'frequency to record')
flags.DEFINE_integer('num_epochs',  30, 'total number of training epochs')
flags.DEFINE_integer('batch_size',  6, 'batch size')
flags.DEFINE_integer('lr_step_size', 30, 'step size to adapt learning rate (piecewise)')
flags.DEFINE_float('lr_decay',       5, 'decay in each step, for 1e-4 and decay 0.1, next steps are 1e-5 and 1e-6')

flags.DEFINE_integer('val_num_per_epoch', 2, 'validate how many times during each epoch')
# Model-related
flags.DEFINE_list('feed_size',      [192, 640], '[height, width] of input image')
flags.DEFINE_bool('do_augmentation',    True, 'apply image augmentation')
flags.DEFINE_float('min_depth',     0.1, 'minimum depth when applying scaling/normalizing to depth estimates')
flags.DEFINE_float('max_depth',     100., 'maximum depth when applying scaling/normalizing to depth estimates')

# Evaluation
flags.DEFINE_float('pred_depth_scale_factor', 1., 'additional depth scaling factor')


def get_options():
    return FLAGS
