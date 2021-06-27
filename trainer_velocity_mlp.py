import tensorflow as tf
import os
from datasets.data_loader_velocitymlp import DataLoader as DataLoader_Velo
from datasets.dataset_kitti import VeloChallenge
from src.detection import Detection
import numpy as np
from tensorflow.keras.layers import MaxPool2D
from models.motion_mlp import FullMotionMLP
from models.distance_mlp import DistanceMLP
from utils import crop_to_aspect_ratio, arrange_display_images
from tqdm import tqdm
import cv2 as cv
from src.trainer_helper import colorize
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import mean_squared_error as calc_mse

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class TrainerMotion:
    def __init__(self, opt):
        self.opt = opt
        self.feed_size = list(map(int, self.opt.feed_size))
        self.train_loader = None
        self.train_iter = None
        self.val_loader = None
        self.val_iter = None
        self.global_step = tf.constant(0, dtype=tf.int64)
        self.depth_models = {}
        self.summary_writer = {}
        self.val_loss_min = tf.constant(1e5, dtype=tf.float32)  # random large value
        self.train_loss_min = tf.constant(1e5, dtype=tf.float32)  # random large value
        self.val_loss_improves = []

        # boundaries = [self.opt.lr_step_size, self.opt.lr_step_size*2]
        # values = [self.opt.learning_rate / scale for scale in [1, self.opt.lr_decay, self.opt.lr_decay**2]]
        # self.lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        # start_lr = learning_rate=self.lr_fn(1+self.opt.start_epoch)
        self.optimizer = tf.keras.optimizers.Adam()

        self.motion_mlp = FullMotionMLP()
        self.num_input_units = 13*13*2+6*2+1
        # pp_x = 675.6 originally
        self.cam_params = {
            'focal_x': 714.2, 'focal_y': 710.4,
            'pp_x': 713.9, 'pp_y': 376.3,
            'height': 1.8}
        self.actual_fps = None

        self.init_app()

    def init_app(self):
        if self.opt.learning_rate is not None:
            self.optimizer.lr.assign(self.opt.learning_rate)
        if not self.opt.debug_mode:
            self.create_summary()
        print('-> create_summary')
        self.build_dataloader_velo()
        print('-> create data loader')
        self.build_depth_models()
        self.build_mlp()

    def create_summary(self):
        if self.opt.recording and not self.opt.debug_mode:
            train_log_dir = os.path.join(self.opt.record_summary_path, self.opt.model_name, 'train')
            self.summary_writer['train'] = tf.summary.create_file_writer(train_log_dir)
            val_log__dir = os.path.join(self.opt.record_summary_path, self.opt.model_name, 'val')
            self.summary_writer['val'] = tf.summary.create_file_writer(val_log__dir)
            print('\tSummary will be stored in %s ' % train_log_dir)

    def build_dataloader_velo(self):
        split_folder = os.path.join('splits', 'velocity_challenge').replace('\\', '/')
        train_file = 'train_files_full.txt'
        val_file = 'val_files.txt'
        data_path = self.opt.data_path
        train_dataset = VeloChallenge(split_folder, train_file)
        val_dataset = VeloChallenge(split_folder, val_file)
        self.actual_fps = train_dataset.fps / self.opt.frame_interval

        if data_path is not None:
            train_dataset.data_path = data_path
            val_dataset.data_path = data_path

        self.train_loader = DataLoader_Velo(train_dataset, opt=self.opt)
        self.train_iter = self.train_loader.build_train_dataset(buffer_size=800)
        self.val_loader = DataLoader_Velo(val_dataset, opt=self.opt)
        self.val_iter = self.val_loader.build_val_dataset(buffer_size=50, shuffle=False)

    def build_depth_models(self):
        encoder_path = os.path.join(self.opt.depth_estimator_path, 'depth_enc')
        decoder_path = os.path.join(self.opt.depth_estimator_path, 'depth_dec')
        self.depth_models['depth_enc'] = tf.saved_model.load(encoder_path)
        self.depth_models['depth_dec'] = tf.saved_model.load(decoder_path)

    def build_mlp(self):
        print("-> Building Full-Motion MLP")
        dummy_inp = tf.random.uniform((self.opt.batch_size, self.num_input_units))
        self.motion_mlp(dummy_inp)
        if self.opt.mlp_weight_path is not None:
            self.motion_mlp.load_weights(self.opt.mlp_weight_path)

    def train(self):
        for epoch in range(self.opt.num_epochs):
            # start from 1, learning rate 15:1e-4; >16:1e-5
            actual_epoch = self.opt.start_epoch + epoch
            # self.optimizer.lr = self.lr_fn(actual_epoch)
            print("\tlearning rate - epoch %d: " % (epoch+1), self.optimizer.get_config()['learning_rate'])

            self.run_epoch(epoch)

            print('validating after epoch %d' % actual_epoch)
            val_loss = self.validate(actual_epoch)
            self.save_model(val_loss, actual_epoch)
            self.adjust_lr()

    def adjust_lr(self):
        """Adjust learning rate based on validation loss"""
        # if latest three epoch don't see improvement, decay
        if len(self.val_loss_improves) > self.opt.lr_step_size:
            if 1 not in self.val_loss_improves[-self.opt.lr_step_size:]:
                curr_lr = self.optimizer.get_config()['learning_rate']
                self.optimizer.lr.assign(curr_lr / self.opt.lr_decay)

    def run_epoch(self, epoch):
        train_loss = []
        for _ in tqdm(range(self.train_loader.steps_per_epoch),
                      desc='%d/%d' % (epoch + 1, self.opt.num_epochs)):
            batch = self.train_iter.get_next()

            trainable_weights = self.motion_mlp.trainable_weights
            grads, loss = self.grad(batch, trainable_weights)

            train_loss.append(loss)
            if self.is_time_to('log', self.global_step):
                batch = self.batch_processor_velo(batch)
                train_loss_mean = tf.reduce_mean(train_loss)
                print("latest train loss", train_loss_mean.numpy())
                self.collect_summary('train', batch[0], train_loss_mean, self.global_step)

            self.optimizer.apply_gradients(zip(grads, trainable_weights))
            self.global_step += 1

    # @tf.function    # turn off to debug, e.g. with plt
    def grad(self, batch, trainables):
        with tf.GradientTape() as tape:
            batch = self.batch_processor_velo(batch)
            total_loss = self.compute_loss(batch, is_train=True)
            grads = tape.gradient(total_loss, trainables)
        return grads, total_loss

    def compute_loss(self, batch, is_train):
        image_stack, bboxes_pairs, gt_motions, valid_nums = batch
        depth_input, offset_y = self.prepare_image_for_depth_estimation(image_stack)
        depth_map_stack, disp_map_stack = self.calc_depth_map(depth_input)
        # print(bboxes_pairs)
        depth_map_pairs = tf.stack(
            tf.split(depth_map_stack, 2, axis=0), axis=1)    # B,2,H,W,3
        # disp_map_pairs = tf.stack(tf.split(disp_map_stack, 2, axis=0), axis=1)    # B,2,H,W,3

        # image_pairs = tf.stack(tf.split(image_stack, 2, axis=0), axis=1)            # B,2,H,W,3
        # print(depth_map_pairs.shape)
        #
        # inp_col1 = [
        #     depth_input[0],
        #     depth_input[6]
        # ]
        # inp_col1 = [
        #     disp_map_pairs[0, 0],
        #     disp_map_pairs[0, 1]
        # ]
        # arrange_display_images(inp_col1)
        batch_size = depth_map_pairs.shape[0]
        loss_per_obj = []
        for i in range(batch_size):
            depth_map_pair = depth_map_pairs[i]
            bboxes_pair = bboxes_pairs[i]
            gt_motion = gt_motions[i]
            valid_num = valid_nums[i]
            bboxes_pair_valid = [bboxes[:valid_num] for bboxes in bboxes_pair]

            # with open("../../motion_ds_experiment/res/depth_map_and_bbox.pkl", 'wb') as df:
            #     pickle.dump([depth_map_pair, bboxes_pair_valid], df)

            feature_vec_pairs, dets_pair = self.get_aggreated_features(depth_map_pair, bboxes_pair_valid)
            gt_motion = gt_motion[:valid_num]

            feature_vec_per_obj, gt_motion = self.add_random_adjustment(feature_vec_pairs, gt_motion, is_train)
            motion_preds = self.derive_motion(feature_vec_per_obj)
            # disp_map_pair = disp_map_pairs[i]
            # dets_prev, dets_curr = dets_pair
            # disp_map_p = disp_map_pair[0].numpy()
            # disp_map_c = disp_map_pair[1].numpy()
            # for det_p, det_c in zip(dets_prev, dets_curr):
            #     l, t, r, b = det_p.offset_ltrb
            #     cv.rectangle(disp_map_p, (l, t), (r, b), (255,255,255))
            #     l, t, r, b = det_c.offset_ltrb
            #     cv.rectangle(disp_map_c, (l, t), (r, b),(200,200,200))
            # arrange_display_images([disp_map_p], [disp_map_c])
            # motion_preds = tf.stack(motion_data_per_obj)
            loss_per_batch = self.mse_loss(gt_motion, motion_preds)
            # print("valid_num", valid_num.numpy())
            # print("motion:\n", [motion.numpy() for motion in motion_preds])
            # print("gt:\n", gt_motion[:valid_num])
            # print("loss \n", loss_per_batch[:valid_num])
            # exit(-1)
            for j in range(valid_num):
                loss_per_obj.append(loss_per_batch[j])

        total_loss = tf.stack(loss_per_obj)
        total_loss = tf.reduce_mean(total_loss)
        return total_loss

    def derive_motion(self, feature_vector_per_obj):
        num = feature_vector_per_obj.shape[0]

        # with open('outputs/feature_vec_all_0.pkl', 'rb') as df:
        #     feature_vector_all = pickle.load(df)
        # with open('outputs/feature_vec_all_0.pkl', 'wb') as df:
        #     pickle.dump(feature_vector_all.numpy(), df)
        # exit(-1)
        motion_data_per_obj = tf.split(
            self.motion_mlp(feature_vector_per_obj), num
        )
        motion_preds = tf.concat(motion_data_per_obj, axis=0)
        return motion_preds

    def add_random_adjustment(self, feature_vec_pairs, gt_motion, is_train=False):
        """Make adjustment on velocity during training
        make_still_case: create a no-movement case, where gt-velocity should be [0., 0.]
        adjust_fps: if not make_still_case, adjust fps, where gt-velocity should be v *= factor.

        These two constraints enforces constraints/hints on inter-connection between inputs
        """
        feature_vec_prev, feature_vec_curr = feature_vec_pairs
        feature_vec_combined = []
        gt_motion_modified = []

        switch_order = tf.random.uniform(shape=(), minval=0., maxval=1.) < self.opt.prob_switch_order
        is_still = tf.random.uniform(shape=(), minval=0., maxval=1.) < self.opt.prob_still_case
        adjust_fps = tf.random.uniform(shape=(), minval=0., maxval=1.) < self.opt.prob_adjust_fps
        fps_factor = tf.random.uniform((), minval=0.5, maxval=2)

        for gt, vec_prev, vec_curr in zip(gt_motion, feature_vec_prev, feature_vec_curr):
            combined_vec = tf.concat([vec_prev, vec_curr], axis=1)  # original

            if is_train and self.opt.motion_type == 'full_motion':
                distance = gt[:2]
                velocity = gt[2:]
                if switch_order:
                    distance, velocity, combined_vec = self.do_switch_order(distance, velocity, combined_vec)
                if is_still:
                    velocity, combined_vec = self.do_still_case(combined_vec)
                if adjust_fps:
                    velocity, combined_vec = self.do_adjust_fps(fps_factor, velocity, combined_vec)
                gt = tf.concat([distance, velocity], axis=0)

            feature_vec_combined.append(combined_vec)
            gt_motion_modified.append(gt)
        feature_vec_combined = tf.concat(feature_vec_combined, axis=0)
        gt_motion = tf.stack(gt_motion_modified)
        return feature_vec_combined, gt_motion

    def decomposite_combined_vec(self, combined_vec):
        fps = combined_vec[:, 0:1]
        vec_p, vec_c = tf.split(combined_vec[:, 1:], 2, axis=1)
        assert vec_c.shape[1] == vec_p.shape[1]
        return fps, vec_p, vec_c

    def do_switch_order(self, distance, velocity, combined_vec):
        """Switch temporal order
        - Distance and Velocity in GroundTruth need to be changed.
        - Vectors for previous and current are switched
        MUST APPLY FIRST
        """
        fps, vec_p, vec_c = self.decomposite_combined_vec(combined_vec)
        distance -= velocity / fps[0]
        velocity *= -1
        combined_vec = tf.concat([fps, vec_c, vec_p], axis=1)   # switched here
        return distance, velocity, combined_vec

    def do_still_case(self, combined_vec):
        """Use identical depth-vec to create a still case
        - GT-Velocity becomes 0
        - Current-vector is duplicated
        In order not to confuse with `do_switch_order`, always duplicate the `current` one, i.e. the 2nd vec
        """
        velocity = tf.zeros(shape=(2,), dtype=tf.float32)
        fps, vec_p, vec_c = self.decomposite_combined_vec(combined_vec)
        combined_vec = tf.concat([fps, vec_c, vec_c], axis=1)
        return velocity, combined_vec

    def do_adjust_fps(self, fps_factor, velocity, combined_vec):
        """Adjust FPS
        - GT-velocity is changed
        - FPS in combined_vec is changed
        """
        fps, vec_p, vec_c = self.decomposite_combined_vec(combined_vec)
        fps *= fps_factor
        velocity *= fps_factor
        combined_vec = tf.concat([fps, vec_p, vec_c], axis=1)
        return velocity, combined_vec

    def mse_loss(self, gt, pred):
        # gt_dist_y, gt_dist_x = tf.split(gt[..., :2], 2, axis=1)
        # gt_velo_y, gt_velo_x = tf.split(gt[..., 2:], 2, axis=1)
        #
        # pred_dist_y, pred_dist_x = tf.split(pred[..., :2], 2, axis=1)
        # pred_velo_y, pred_velo_x = tf.split(pred[..., 2:], 2, axis=1)
        #
        # dist_err_y = calc_mse(gt_dist_y, pred_dist_y)
        # dist_err_x = calc_mse(gt_dist_x, pred_dist_x) * self.opt.side_w_scale   # give more attention to side-movement
        #
        # velo_err_y = calc_mse(gt_velo_y, pred_velo_y)
        # velo_err_x = calc_mse(gt_velo_x, pred_velo_x) * self.opt.side_w_scale
        #
        # total_loss = ((dist_err_x + dist_err_y) * self.opt.dist_loss_w + \
        #              (velo_err_x + velo_err_y) * self.opt.velo_loss_w) /2
        gt_dist, gt_velo = tf.split(gt, 2, axis=1)
        pred_dist, pred_velo = tf.split(pred, 2, axis=1)

        dist_err = calc_mse(gt_dist, pred_dist)
        velo_err = calc_mse(gt_velo, pred_velo)

        total_loss = dist_err * self.opt.dist_loss_w + velo_err * self.opt.velo_loss_w
        return total_loss

    def validate(self, epoch):
        loss_mean = []
        collect_imgs = None
        for _ in tqdm(range(self.val_loader.steps_per_epoch),
                      desc='validate: %d/%d' % (epoch - self.opt.start_epoch + 1, self.opt.num_epochs)):
            batch = self.val_iter.get_next()
            batch = self.batch_processor_velo(batch)
            total_loss = self.compute_loss(batch, is_train=False)

            loss_mean.append(total_loss)
            collect_imgs = batch[0]
        val_loss = tf.reduce_mean(loss_mean)
        self.collect_summary('val', collect_imgs, val_loss, global_step=self.global_step)
        return val_loss

    def save_model(self, val_loss, epoch):
        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            self.val_loss_improves.append(1)
            weights_name = "velocity_mlp_e{:03d}_{:.2f}.h5".format(epoch, val_loss)
            if not self.opt.debug_mode:
                weights_path = os.path.join(self.opt.save_model_path, weights_name)
                if not os.path.isdir(self.opt.save_model_path):
                    os.makedirs(self.opt.save_model_path)
                self.motion_mlp.save_weights(weights_path)
            print("saving weights with lowest val_loss {:.2f} to:".format(val_loss), weights_path)
        else:
            self.val_loss_improves.append(0)
            print("current loss doesn't improve ({:.2f} Vs. {:.2f}), skip saving model".format(self.val_loss_min, val_loss))

    def calc_depth_map(self, image_stack):
        """Compute depth maps for all images
        Args:
            image_stack: Tensor
                shape: (2*B,H,W,3)
        """
        encoder = self.depth_models['depth_enc'].signatures['serving_default']
        decoder = self.depth_models['depth_dec'].signatures['serving_default']
        features_raw = encoder(image_stack)
        features = {}
        for i in range(1, 6):
            features['input_%d' % i] = (features_raw['output_%d' % i])
        pred_disp = decoder(**features)
        for k, v in pred_disp.items():
            disp_map_stack = v
        disp_map_stack, depth_map_stack = self.disp_to_depth(disp_map_stack)
        # inp1 = [colorize(disp_map_stack[0], cmap='plasma')]
        # arrange_display_images(inp1)
        return depth_map_stack, disp_map_stack

    def get_aggreated_features(self, depth_map_pair, bboxes_pair):
        """Generate feature vector from previous and current depth map and geometry parameter
        It only applies to a simple case:
        -> All current bboxes have corresponding bbox in the previous frame (-1 or -2).
        This is for VelocityChallenge dataset, where tracked bboxes are pre-computed

        Args:
            depth_map_pair: Tensor
                pair of depth / disp map, in temporal order, shape (2, H, W, 1)
            bboxes_pair: Tensor
                pair of Detection, in temporal order. Shape (2, N, 4)
        """
        dets_pair = []
        for bboxes in bboxes_pair:
            dets_pair.append([Detection(ltwh=bbox) for bbox in bboxes])

        dets_pair = self.associate_IDs(dets_pair)
        dets_pair = self.calc_geometry_clues(dets_pair, self.cam_params)
        dets_pair = self.extract_depth_feature_pair(depth_map_pair, dets_pair)
        feature_vec_pairs = self.aggregate_geo_clues_and_depth_feature(dets_pair)
        return feature_vec_pairs, dets_pair

    def aggregate_geo_clues_and_depth_feature(self, dets_neighbor_frames):
        """Arggregate geometric clues and detph feature into vector
        The aggregated feature vector of two neighbor frames are concatenated together as one final vector
        Args:
            dets_neighbor_frames: list
                contains previous and current list of Detections
        Returns:
            feature_vec_combined: list of Tensor
                [geo_prev, depth_prev, geo_curr, depth_curr], dimension:
                [6, 13*13, 6, 13*13] -> [1, 350]
                size of depth_feature_vector, i.e. 13, might be adjusted.
        """
        dets_prev, dets_curr = dets_neighbor_frames
        fps = tf.reshape(tf.constant([self.actual_fps], dtype=tf.float32), (1, -1))
        feature_vec_prev = []
        for det in dets_prev:
            fused_feature = tf.concat([
                fps,
                tf.reshape(det.get_geo_clues(), (1, -1)),
                tf.reshape(det.get_depth_feature(), (1, -1))
            ], axis=1)
            feature_vec_prev.append(fused_feature)

        feature_vec_curr = []
        for det in dets_curr:
            fused_feature = tf.concat([
                tf.reshape(det.get_geo_clues(), (1, -1)),
                tf.reshape(det.get_depth_feature(), (1, -1))
            ], axis=1)
            feature_vec_curr.append(fused_feature)

        return feature_vec_prev, feature_vec_curr

    def associate_IDs(self, dets_latest_frames):
        """Associate ID to bboxes in different frames
        It is a simple substitute to the DeepSort.
        Since we consider the simplest case, where the same ID is assigned according to minimal difference

        Args:
            dets_latest_frames: list
                [[Detection, ...], ...]
                Each member stores detections from ONE image of the frame. The last is the newest.
                This list will be maintained in a fixed length, so we always take the 1st and the last member.
        Returns:
            dets_neighbor_frames: list
                [[previous Detections], [current Detections]]]
                The two lists are alligned by ID order.
        """
        dets_prev, dets_curr = dets_latest_frames
        boxes_prev = np.stack([det.to_ltrb() for det in dets_prev])
        boxes_curr = [det.to_ltrb() for det in dets_curr]
        dets_prev_ordered = []
        for ID, box_curr in enumerate(boxes_curr):
            diff = np.sum(np.abs(np.subtract(boxes_prev, box_curr)), axis=1)
            idx = np.argmin(diff)

            dets_curr[ID].assign_ID(ID)
            dets_prev[idx].assign_ID(ID)
            dets_prev_ordered.append(dets_prev[idx])

        dets_neighbor_frames = [dets_prev_ordered, dets_curr]
        return dets_neighbor_frames

    def calc_geometry_clues(self, dets_neighbor_frames, cam_params):
        """assign geometry clues in each Detection"""

        def calc_(det, cam_params):
            pp_x, pp_y = cam_params['pp_x'], cam_params['pp_y']
            fx, fy = cam_params['focal_x'], cam_params['focal_y']
            l, t, r, b = det.to_ltrb()
            geo_clues = [
                fx / (r - l),
                fy / (b - t),
                (l - pp_x) / fx,
                (r - pp_x) / fx,
                (t - pp_y) / fx,
                (b - pp_y) / fx
            ]
            return geo_clues

        for det in dets_neighbor_frames[0]:
            geo_clues = calc_(det, cam_params)
            det.receive_geo_clues(
                np.array(geo_clues, dtype=np.float32)
            )
        for det in dets_neighbor_frames[1]:
            geo_clues = calc_(det, cam_params)
            det.receive_geo_clues(
                np.array(geo_clues, dtype=np.float32)
            )
        return dets_neighbor_frames

    def extract_depth_feature_pair(self, depth_maps, dets_neighbor_frames):
        """From depth map patches get feature vector
        Args:
            depth_maps: Tensor
                (2, 192, 640, 1), two depth maps
            dets_neighbor_frames: list of Detection
                Detections for two neighbor frames, must make sure:
                - the IDs between boxes of two frames are one-to-one associated
                - the list is ordered by ID, therefore it's ok to directly unpack
        Returns:
            list of feature vectors
        """

        def extract_vec_per_bbox(depth_map, bbox_ltrb, pool_out_size=13):
            assert len(depth_map.shape) in (3, 4) and depth_map.shape[-1] == 1, \
                "depth_map should have shape (1,192,640,1) or (192, 640, 1), got {}".format(depth_map.shape)
            if len(depth_map.shape) == 3:
                depth_map = np.expand_dims(depth_map, 0)
            stride, kernel_size = 2, 3
            l, t, r, b = bbox_ltrb
            resize_to = pool_out_size * stride + int(kernel_size // 2)

            feature_patch = depth_map[:, t:b, l:r, :]
            # print("patch size", feature_patch.shape)
            feature_patch = tf.image.resize(feature_patch, (resize_to, resize_to))
            feature_patch = MaxPool2D((kernel_size, kernel_size), strides=stride)(feature_patch)
            feature_vec = tf.reshape(feature_patch, (-1))
            # if self.opt.debug_mode:
            #     inp1 = [depth_map[0]]
            #     cv.rectangle(inp1[0], (l, t), (r, b), (255, 255, 255))
            #     inp2 = [feature_patch[0]]
            #     arrange_display_images(inp1, inp2)
            return feature_vec

        def offset_and_rescale_bbox(det, offset_y=168, image_scaling=2.):
            """offset and resize bbox
            The depthnet's output shape is (192, 640), the bbox is generated in original resolution (720, 1280)
            we need to center-crop and offset the bbox, as we did to depthnet's input
            Args:
                det: Detection object
                offset_y: int
                image_scaling: float ot int
                    original width / feed width of monodepth2 = 1280/640
            Returns:
                ltrb: list of int
            """
            skip, ltrb = det.to_offset_ltrb(offset_y, boundaries=(0, 192 * image_scaling))
            ltrb = det.rescale_bbox(ltrb, image_scaling, as_int=True)
            ltrb[0] = max(0, ltrb[0])
            ltrb[2] = min(640, ltrb[2])
            return ltrb

        def resize_map2vec_all_bboxes(dets, depth_map):
            for det in dets:
                bbox_ltrb = offset_and_rescale_bbox(det)
                # det.offset_ltrb = bbox_ltrb  # debug
                det.receive_depth_feature(
                    extract_vec_per_bbox(depth_map, bbox_ltrb)
                )
            return dets

        depth_map_prev, depth_map_curr = depth_maps
        dets_prev, dets_curr = dets_neighbor_frames
        dets_prev = resize_map2vec_all_bboxes(dets_prev, depth_map_prev)
        dets_curr = resize_map2vec_all_bboxes(dets_curr, depth_map_curr)
        dets_neighbor_frames = [dets_prev, dets_curr]
        return dets_neighbor_frames

    def batch_processor_velo(self, batch):
        image_pairs, bboxes_pairs, gt_motions, valid_nums = batch
        image_pairs = tf.concat(tf.split(image_pairs, 2, axis=-1), axis=0)   # 2*B,H,W,3
        gt_motions = gt_motions[:, 0, ...]
        return image_pairs, bboxes_pairs, gt_motions, valid_nums

    def disp_to_depth(self, disp, flatten=False, base_scaling=1.):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / self.opt.max_depth
        max_disp = 1 / self.opt.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        if flatten:
            scaled_disp = tf.reshape(scaled_disp, [-1])
            depth = tf.reshape(depth, [-1])
        if base_scaling != 1:
            depth *= base_scaling
        return scaled_disp, depth

    def collect_summary(self, mode, input_imgs, loss, global_step):
        """collect summary for train / validation
        Args:
            mode: str
                "train" or "val"
            loss: tf.constant
            input_imgs: Tensor
                shape in (2*B, H, W, 3)
            global_step: tf.constant
        """
        writer = self.summary_writer[mode]
        batch_size = int(input_imgs.shape[0] / 2)
        num = min(2, batch_size)
        with writer.as_default():
            tf.summary.scalar('total_loss', loss, step=global_step)
            for i in range(num):
                image_pair = tf.concat([input_imgs[i:i+1], input_imgs[i+batch_size: i+batch_size+1]], axis=0)
                tf.summary.image('image_pair%d' % i, image_pair, step=global_step)
        writer.flush()

    def is_time_to(self, event, global_step):
        if self.opt.debug_mode:
            return False

        is_time = False
        events = {0: 'log', 1: 'validate', 2: 'special_pass'}
        if event not in events.values():
            raise NotImplementedError

        if event == events[0]:
            early_phase = global_step % self.opt.record_freq == 0 and global_step < 1000
            late_phase = global_step % (self.opt.record_freq*2) == 0
            if early_phase or late_phase:
                is_time = True
        elif event == events[1]:
            special_pass = global_step % 5 == 0 and self.opt.debug_mode
            normal_pass = global_step % (self.train_loader.steps_per_epoch // self.opt.val_num_per_epoch) == 0
            is_time = normal_pass or special_pass
        elif event == events[2]:
            is_time = False
        return is_time

    def prepare_image_for_depth_estimation(self, images):
        """Preprocess image for depth network"""
        depth_input = []
        offset_y = 0
        for i in range(images.shape[0]):
            image = images[i]
            image_cropped, offset_y = crop_to_aspect_ratio(image, self.feed_size)
            # image_cropped = cv.resize(image_cropped.numpy(), (640, 192))
            depth_input.append(image_cropped)
        depth_input = tf.stack(depth_input)
        depth_input = tf.image.resize(depth_input, self.feed_size)

        # with open(r'D:\MA\Recources\Monodepth2-tf2_stable\outputs\aug_image.pkl', 'rb') as df:
        #     imported = pickle.load(df)
        # for i in range(depth_input.shape[0]):
        #     image_ = depth_input[i]
        #     print("diff:", imported.shape, image_.shape, tf.reduce_sum(tf.squeeze(imported) - tf.squeeze(image_)))
        # image_ = depth_input[6]
        # plt.imshow(image_), plt.show()
        # plt.imshow(np.squeeze(imported)), plt.show()

        return depth_input, offset_y


class TrainerDistance(TrainerMotion):
    def __init__(self, opt):
        super(TrainerDistance, self).__init__(opt)
        self.motion_mlp = DistanceMLP()
        self.num_input_units = 13*13+6

    def build_mlp(self,):
        print("-> Building Distance-Only MLP")
        dummy_inp = tf.random.uniform((self.opt.batch_size, self.num_input_units))
        self.motion_mlp(dummy_inp)
        if self.opt.mlp_weight_path is not None:
            self.motion_mlp.load_weights(self.opt.mlp_weight_path)

    def derive_motion(self, feature_pair_per_obj):
        motion_data_per_obj = []
        for feature_pair in feature_pair_per_obj:
            inps = tf.stack(tf.split(feature_pair, 2))          # 350, -> 2,175
            dists = self.motion_mlp(inps)                       # 2,2
            dist_p, dist_c = tf.split(dists, 2)         # 2 * (1,2)
            velo_c = (dist_c - dist_p) * self.actual_fps
            motion_data_per_obj.append(
                tf.squeeze(tf.concat([dist_c, velo_c], axis=1))     # (4,)
            )
        motion_data_per_obj = tf.stack(motion_data_per_obj)
        return motion_data_per_obj

    def aggregate_geo_clues_and_depth_feature(self, dets_neighbor_frames):
        """Arggregate geometric clues and detph feature into vector
        The aggregated feature vector of two neighbor frames are concatenated together as one final vector
        Args:
            dets_neighbor_frames: list
                contains previous and current list of Detections
        Returns:
            feature_vec_combined: list of Tensor
                [geo_prev, depth_prev, geo_curr, depth_curr], dimension:
                [6, 13*13, 6, 13*13] -> [1, 350]
                size of depth_feature_vector, i.e. 13, might be adjusted.
        """
        dets_prev, dets_curr = dets_neighbor_frames
        feature_vec_prev = []
        for det in dets_prev:
            fused_feature = tf.concat([
                tf.reshape(det.get_geo_clues(), (1, -1)),
                tf.reshape(det.get_depth_feature(), (1, -1))
            ], axis=1)
            feature_vec_prev.append(fused_feature)

        feature_vec_curr = []
        for det in dets_curr:
            fused_feature = tf.concat([
                tf.reshape(det.get_geo_clues(), (1, -1)),
                tf.reshape(det.get_depth_feature(), (1, -1))
            ], axis=1)
            feature_vec_curr.append(fused_feature)
        # feature_vec_combined = [tf.concat([vec_prev, vec_curr], axis=1) for vec_prev, vec_curr
        #                         in zip(feature_vec_prev, feature_vec_curr)]
        return feature_vec_prev, feature_vec_curr


def verbose_flags(opt):
    print("==================================")
    print("motion_type\t", opt.motion_type)
    print("learning_rate\t", opt.learning_rate)
    print("frame_interval\t", opt.frame_interval)
    print("start_epoch\t", opt.start_epoch)
    print("dist_loss_w\t", opt.dist_loss_w)
    print("velo_loss_w\t", opt.velo_loss_w)
    print("save_model_path\t", opt.save_model_path)
    print("save_root\t", opt.save_root)
    print("data_path\t", opt.data_path)
    print("bbox_folder\t", opt.bbox_folder)
    print("recording\t", opt.recording)
    print("record_path\t", opt.record_summary_path)
    print("batch_size\t", opt.batch_size)
    print("num_epochs\t", opt.num_epochs)
    print("lr_step_size\t", opt.lr_step_size)
    print("lr_decay\t", opt.lr_decay)
    print("==================================")


def main(_):
    opt = get_options()
    verbose_flags(opt)
    if opt.motion_type == 'full_motion':
        trainer_velo = TrainerMotion(opt)
        trainer_velo.train()
    elif opt.motion_type == 'distance_only':
        trainer_dist = TrainerDistance(opt)
        trainer_dist.train()
        raise NotImplementedError


if __name__ == '__main__':
    from options_motion_mlp import get_options
    from absl import app
    app.run(main)
