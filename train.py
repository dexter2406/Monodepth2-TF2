from absl import app, flags
import datetime
from new_trainer_v1 import Trainer
import os
import warnings

rootdir = os.path.dirname(__file__)
curren_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
all_models = ['depth_enc', 'depth_dec', 'pose_enc', 'pose_enc']

FLAGS = flags.FLAGS
flags.DEFINE_string('weights_dir', './logs/weights/20210419-172132',  'the folder that stores weights files.')
flags.DEFINE_list('models_to_load', all_models,
                  'load weights for specified models, by default all of them')

flags.DEFINE_string('model_name', curren_time, 'specify a dirname to collect weights, if not, current time is used')
flags.DEFINE_bool('train_depth', True, 'whether to train depth decoder-encoder')
flags.DEFINE_bool('train_pose', True, 'whether to train pose decoder-encoder')
flags.DEFINE_bool('from_scratch', False, 'whether trained from scratch, coorperate with load_weights_folder')
flags.DEFINE_string('save_model_path', '', 'path where weights are saved')

flags.DEFINE_string('run_mode', 'train', 'train or eval')
flags.DEFINE_bool('recording', False, 'whether to write results by tf.summary')
flags.DEFINE_string('record_summary_path', 'logs/gradient_tape/', 'root path to write summary')
flags.DEFINE_integer('record_freq', 250, 'frequency to record')
flags.DEFINE_integer('num_epochs', 10, 'total number of training epochs')
flags.DEFINE_integer('batch_size', 6, 'batch size')
flags.DEFINE_bool('debug_mode', False, 'inspect intermediate results')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate')
flags.DEFINE_integer('lr_step_size', 15, 'step size to adapt learning rate (piecewise)')

flags.DEFINE_integer('num_scales', 4, 'number of scales')
flags.DEFINE_integer('src_scale', 0, 'source scale')
flags.DEFINE_integer('height', 192, 'height of input image')
flags.DEFINE_integer('width', 640, 'width of input image')
flags.DEFINE_list('frame_idx', [0, -1, 1], 'index of target, previous and next frame')
flags.DEFINE_bool('do_augmentation', True, 'apply image augmentation')
flags.DEFINE_bool('do_automasking', True, 'apply auto masking')

flags.DEFINE_float('ssim_ratio', 0.85, 'ratio to calculate SSIM loss')
flags.DEFINE_float('smoothness_ratio', 1e-3, 'ratio to calculate smoothness loss')
flags.DEFINE_float('min_depth', 0.1, 'minimum depth when applying scaling/normalizing to depth estimates')
flags.DEFINE_float('max_depth', 100., 'maximum depth when applying scaling/normalizing to depth estimates')


def check_options(FLAGS, debug=False):
    print("-> Check options...")
    for m in FLAGS.models_to_load:
        if m not in all_models:
            raise ValueError("\t'%s' is not supported model, choose from: " % m, all_models)

    FLAGS.debug_mode = debug
    if FLAGS.debug_mode:
        print('\t[debug mode] Check intermediate results. '
              'Models will not be trained or saved even if options are given.')
        FLAGS.batch_size = 1
        print("\t[debug mode] batch_size is reset to %d\n" % FLAGS.batch_size)

    else:
        print("\tTURN ON @tf.function for grad() and DataProcessor!")
        save_dir = FLAGS.save_model_path
        if FLAGS.save_model_path == '':
            save_dir = os.path.join(rootdir, 'logs/weights/')
            print('\tno save_model_path specified, use %s instead' % FLAGS.save_model_path)

        save_path = os.path.join(save_dir, FLAGS.model_name)
        if not os.path.isdir(save_path):
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
                raise ValueError('\tif not from scratch, please specify --weights_dir to load weights')
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


def start_train(FLAGS):
    trainer = Trainer(FLAGS)
    trainer.train()


def main(_):
    check_options(FLAGS, debug=True)
    start_train(FLAGS)


if __name__ == '__main__':
    app.run(main)
