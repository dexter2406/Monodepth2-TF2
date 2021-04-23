import numpy as np
from utils import readlines, disp_to_depth
import os
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from src.trainer_helper import build_models

from datasets.dataset_kitti import KITTIRaw, KITTIOdom
from datasets.data_loader_kitti import DataLoader
from src.DataPprocessor import DataProcessor
from src.eval_helper import compute_errors
from options import get_options
import cv2 as cv
from absl import app


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
splits_dir = os.path.join(root_dir, "splits")
depth_model_names = ['depth_enc', 'depth_dec']


def evaluate_depth(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if opt.use_ext_res is not None:
        # ----------
        # Prepare models
        # ----------
        assert os.path.isdir(opt.weights_dir), 'folder: %s doesn\'t exists' % opt.weights_dir
        print('-> Loading weights from {}'.format(opt.weights_dir))
        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        models = {'depth_enc': ResNet18_new([2, 2, 2, 2]),
                  'depth_dec': DepthDecoder_full()}
        build_models(models)
        for m_name in depth_model_names:
            m_path = os.path.join(opt.weights_dir, m_name + '.h5')
            models[m_name].load_weights(m_path)

        # ----------
        # Get dataset
        # ----------
        print('-> Preparing dataset: KITTI_Raw...')
        split_folder = os.path.join('splits', opt.eval_split)
        split_name = 'test_files.txt'.format(opt.eval_split)
        path_tmp = os.path.join(split_folder, split_name)
        assert os.path.isfile(path_tmp), '%s is not valid path to split files' % path_tmp

        opt.frame_idx = [0]
        num_scales = 1
        batch_size = 16
        dataset = KITTIRaw(split_folder, split_name, data_path=opt.data_path)
        data_loader = DataLoader(dataset, num_epoch=1,
                                 batch_size=batch_size, frame_idx=opt.frame_idx)
        eval_iter = data_loader.build_eval_dataset()
        batch_processor = DataProcessor(frame_idx=opt.frame_idx, num_scales=num_scales,
                                        intrinsics=dataset.K)

        # ----------
        # Generate predicted disparity map
        # ----------
        print('-> Generate predicted disparity map...')
        pred_disps = []
        output = {}
        for batch in eval_iter:
            # batch = eval_iter.get_next()
            input_imgs, input_Ks = batch_processor.prepare_batch_val(batch)
            input_color = input_imgs[('color', 0, 0)]
            disp_raw = models['depth_dec'](models['depth_enc'](input_color))
            output[('disp', 0)] = disp_raw['output_0']
            pred_disp, _ = disp_to_depth(output[('disp', 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp[..., 0].numpy()   # squeeze the last dim
            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    # todo: use imported disp to eval
    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_dir = os.path.join(root_dir, 'outputs', 'disps')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'disps_{}_split.npy'.format(opt.eval_split))
        print('-> Saving predicted disparities to ', output_path)
        np.save(output_path, pred_disps)

    # Just need generate predictions, but not to evaluate them
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # ----------
    # Get ground truth and start evaluation
    # ----------
    print("-> Loading depth ground truth...")
    gt_path = os.path.join(splits_dir, opt.eval_split, 'gt_depths.npz')
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating...")
    errors = []
    ratios = []
    for i in range(len(pred_disps)):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1. / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]   # tf.boolean_mask
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if opt.use_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.use_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print('\tScaling ratios | median: {:0.3f} | std: {:0.3f}'.format(med, np.std(ratios/med)))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def main_test(_):
    options = get_options()
    evaluate_depth(options)


if __name__ == "__main__":
    app.run(main_test)
