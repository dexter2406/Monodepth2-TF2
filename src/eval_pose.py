import os
import numpy as np
from datasets.dataset_kitti import KITTIOdom
from datasets.data_loader_kitti import DataLoader
from utils import readlines
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from src.eval_helper import dump_xyz, compute_ate
from src.DataPprocessor import DataProcessor
from src.trainer_helper import build_models, transformation_from_parameters
from options import get_options
import tensorflow as tf

root_dir = os.path.dirname(__file__)


def evalualte_pose(opt):
    assert os.path.isdir(opt.weights_dir), 'folder %s doesn\'t exists' % opt.weights_dir

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])
    filenames = readlines(
        os.path.join(root_dir, "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    # ----------
    # Get dataset
    # ----------
    opt.frame_idx = [0, 1]
    split_folder = os.path.join('splits', opt.eval_split)
    split_name = 'test_files{:02d}.txt'.format(int(opt.eval_split[-1]))
    dataset = KITTIOdom(split_folder, split_name, data_path=opt.data_path)
    data_loader = DataLoader(dataset, num_epoch=1,
                             batch_size=2, frame_idx=opt.frame_idx)
    eval_iter = data_loader.build_eval_dataset()
    batch_processor = DataProcessor(frame_idx=opt.frame_idx, intrinsics=dataset.K)

    # ----------
    # Load Models
    # ----------
    models = {
        'pose_enc': ResNet18_new([2, 2, 2, 2]),
        'pose_dec': DepthDecoder_full()
    }
    build_models(models)

    for m_name in models.keys():
        m_path = os.path.join(opt.weights_dir, m_name + '.h5')
        models[m_name].load_weights(m_path)

    # ----------
    # Generate poses
    # ----------
    print("-> Computing pose predictions")
    pred_poses = []

    for i in range(len(filenames)):
        batch = eval_iter.get_next()
        input_imgs, input_Ks = batch_processor.prepare_batch(batch)
        pose_inps = [input_imgs[('color_aug', i, 0)] for i in opt.frame_idx]

        features = models['pose_enc'](tf.concat(pose_inps), axis=3)
        pose_outs = models['pose_dec'](features)
        axisangle, translation = pose_outs['angles'], pose_outs['translations']

        pred = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        pred_poses.append(pred)

    pred_poses = np.concatenate(pred_poses)

    # ----------
    # Load file of ground-truth poses
    # ----------
    gt_poses_dir = os.path.join(root_dir, "outputs", "poses")
    if not os.path.isdir(gt_poses_dir):
        os.makedirs(gt_poses_dir)
    gt_poses_path = os.path.join(gt_poses_dir, "{:02d}.txt".format(sequence_id))

    # ----------
    # Coordinates transormations
    # ----------
    #  - global poses
    gt_global_poses = np.loadtxt(gt_poses_path).reshape((-1, 3, 4))
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]
    #  - local poses
    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    # ----------
    # Compute trajectory errors
    # ----------
    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n\tTrajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(gt_poses_dir, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == '__main__':
    options = get_options()
    evalualte_pose(options)
