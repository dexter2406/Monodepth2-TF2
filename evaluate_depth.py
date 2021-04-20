import numpy as np
from utils import readlines, disp_to_depth
import os
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from src.trainer_helper import build_models
from src.dataset_loader_kitti_raw import DataLoader_KITTI_Raw
from src.DataPprocessor import DataProcessor
from src.eval_helper import compute_errors
from options import get_options
import cv2 as cv


root_dir = os.path.dirname(__file__)
splits_dir = os.path.join(root_dir, "splits")


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if not opt.use_ext_disp:
        # ----------
        # Prepare models
        # ----------
        assert os.path.isdir(opt.weights_dir), 'folder %s doesn\'t exists' % opt.weights_dir
        print('-> Loading weights from {}'.format(opt.weights_dir))
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        models = {}

        models['depth_enc'] = ResNet18_new([2, 2, 2, 2])
        models['depth_dec'] = DepthDecoder_full()
        build_models(models)

        for m_name in ['depth_enc', 'depth_dec']:
            m_path = os.path.join(opt.weights_dir, m_name + '.h5')
            models[m_name].load_weights(m_path)

        # ----------
        # Get dataset
        # ----------
        data_loader = DataLoader_KITTI_Raw(opt.num_epochs, opt.batch_size, debug_mode=opt.debug_mode,
                                           dataset_for='eval', split_name='eigen_zhou')
        dataset = data_loader.build_dataset()
        dataset_iter = iter(dataset)
        batch_processor = DataProcessor()

        # ----------
        # Generate predicted disparity map
        # ----------
        pred_disps = []
        for i in range(len(filenames)):
            batch = dataset_iter.get_next()
            input_imgs, input_Ks = batch_processor.prepare_batch(batch[..., :3], batch[..., 3:])
            input_color = input_imgs[('color', 0, 0)]

            output = models['depth_dec'](models['depth_enc'](input_color))
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp[..., 0].numpy()

            pred_disps.append(pred_disp)

    # use imported disp to eval
    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_dir = os.path.join(root_dir, 'output_disps')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'disps_{}_split.npy'.format(opt.eval_split))
        print('-> Saving predicted disparities to ', output_path)

    # Just need generate predictions, but not to evaluate them
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # ----------
    # Get ground truth and start evaluation
    # ----------
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

        pred_depth = pred_depth[mask]
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


if __name__ == "__main__":
    options = get_options()
    evaluate(options)
