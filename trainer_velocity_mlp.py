import tensorflow as tf
import os
from datasets.data_loader_velocitymlp import DataLoader as DataLoader_Velo
from datasets.dataset_kitti import VeloChallenge
from src.detection import Detection
import numpy as np
from tensorflow.keras.layers import MaxPool2D
from models.velocity_mlp import VelocityMLP
from utils import crop_to_aspect_ratio, arrange_display_images
from tqdm import tqdm


class TrainerVelo:
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

        boundaries = [self.opt.lr_step_size, self.opt.lr_step_size*2]
        values = [self.opt.learning_rate / scale for scale in [1, self.opt.lr_decay, self.opt.lr_decay**2]]
        self.lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_fn(1))

        self.velocity_mlp = VelocityMLP()
        self.cam_params = {
            'focal_x': 714.2, 'focal_y': 710.4,
            'pp_x': 675.6, 'pp_y': 376.3,
            'height': 1.8}
        self.actual_fps = None

        self.init_app()

    def init_app(self):
        self.create_summary()
        print('-> create_summary')
        self.build_dataloader_velo()
        print('-> create data loader')
        self.build_models()
        print('-> build depth model')

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
        self.train_iter = self.train_loader.build_train_dataset(buffer_size=1000)

        self.val_loader = DataLoader_Velo(val_dataset, opt=self.opt)
        self.val_iter = self.val_loader.build_val_dataset(buffer_size=500, shuffle=False)

    def build_models(self):
        encoder_path = os.path.join(self.opt.depth_estimator_path, 'depth_enc')
        decoder_path = os.path.join(self.opt.depth_estimator_path, 'depth_dec')
        self.depth_models['depth_enc'] = tf.saved_model.load(encoder_path)
        self.depth_models['depth_dec'] = tf.saved_model.load(decoder_path)

    def train(self):
        for epoch in range(self.opt.num_epochs):
            self.optimizer.lr = self.lr_fn(epoch+1)     # start from 1, learning rate 15:1e-4; >16:1e-5
            print("\tlearning rate - epoch %d: " % (epoch+1), self.optimizer.get_config()['learning_rate'])

            self.run_epoch(epoch)
            print('validating after epoch %d' % (epoch+1))
            self.validate(epoch)

    def run_epoch(self, epoch):
        for _ in tqdm(range(self.train_loader.steps_per_epoch),
                      desc='Epoch%d/%d' % (epoch + 1, self.opt.num_epochs)):
            batch = self.train_iter.get_next()

            trainable_weights = self.velocity_mlp.trainable_weights
            grads = self.grad(batch, trainable_weights, self.global_step)
            self.optimizer.apply_gradients(zip(grads, trainable_weights))
            self.global_step += 1

    # @tf.function    # turn off to debug, e.g. with plt
    def grad(self, batch, trainables, global_step):
        with tf.GradientTape() as tape:
            batch = self.batch_processor_velo(batch)
            total_loss = self.compute_loss(batch)
            grads = tape.gradient(total_loss, trainables)
            if self.is_time_to('log', global_step):
                self.collect_summary('train', batch[0], total_loss, global_step=global_step)
        return grads

    def compute_loss(self, batch):
        image_stack, bboxes_pairs, gt_motions, valid_nums = batch
        depth_input, offset_y = self.prepare_image_for_depth_estimation(image_stack)
        depth_map_stack = self.calc_depth_map(depth_input)

        depth_map_pairs = tf.stack(tf.split(depth_map_stack, 2, axis=0), axis=1)    # B,2,H,W,3
        # image_pairs = tf.stack(tf.split(image_stack, 2, axis=0), axis=1)            # B,2,H,W,3
        # print(depth_map_pairs.shape)
        #
        # inp_col1 = [
        #     depth_input[0],
        #     depth_input[6]
        # ]
        # inp_col2 = [
        #     depth_map_pairs[0, 0],
        #     depth_map_pairs[0, 1]
        # ]
        # arrange_display_images(inp_col1, inp_col2)
        batch_size = depth_map_pairs.shape[0]
        loss_per_obj = []
        for i in range(batch_size):
            depth_map_pair = depth_map_pairs[i]
            bboxes_pair = bboxes_pairs[i]
            gt_motion = gt_motions[i]
            valid_num = valid_nums[i]
            bboxes_pair_valid = [bboxes[:valid_num] for bboxes in bboxes_pair]
            feature_vector_per_obj = self.get_aggreated_features(depth_map_pair, bboxes_pair_valid)
            motion_data_per_obj = []
            for feat_vec in feature_vector_per_obj:
                motion_data_per_obj.append(
                    tf.squeeze(self.velocity_mlp(feat_vec))
                )
            motion_preds = tf.stack(motion_data_per_obj)
            gt_motion = gt_motion[:valid_num]
            # print("valid_num", valid_num.numpy())
            # print("gt", gt_motion.shape, '\n', gt_motion.numpy())
            # print("pred\n", motion_preds.shape, '\n', motion_preds.numpy())
            loss_per_batch = self.mse_loss(gt_motion, motion_preds)
            for j in range(valid_num):
                loss_per_obj.append(loss_per_batch[j])

        total_loss = tf.stack(loss_per_obj)
        total_loss = tf.reduce_mean(total_loss)
        return total_loss

    def mse_loss(self, gt, pred):
        gt_dist = gt[..., :2]
        gt_velo = gt[..., 2:]
        pred_dist = pred[..., :2]
        pred_velo = pred[..., 2:]

        dist_err = tf.keras.metrics.mean_squared_error(gt_dist, pred_dist)
        velo_err = tf.keras.metrics.mean_squared_error(gt_velo, pred_velo)

        total_loss = dist_err * self.opt.dist_loss_w + velo_err * self.opt.velo_loss_w
        return total_loss

    def validate(self, epoch):
        loss_mean = []
        collect_imgs = None
        for _ in tqdm(range(self.val_loader.steps_per_epoch),
                      desc='validate: epoch%d/%d' % (epoch + 1, self.opt.num_epochs)):
            batch = self.val_iter.get_next()
            batch = self.batch_processor_velo(batch)
            total_loss = self.compute_loss(batch)

            loss_mean.append(total_loss)
            collect_imgs = batch[0]
        self.collect_summary('val', collect_imgs, tf.reduce_mean(loss_mean), global_step=self.global_step)

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
        return depth_map_stack

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
        dets_pair = self.resize_depth_map_pair(depth_map_pair, dets_pair)
        feature_vec = self.aggregate_geo_clues_and_depth_feature(dets_pair, self.actual_fps)
        return feature_vec

    def aggregate_geo_clues_and_depth_feature(self,dets_neighbor_frames, fps):
        """Arggregate geometric clues and detph feature into vector
        The aggregated feature vector of two neighbor frames are concatenated together as one final vector
        Args:
            dets_neighbor_frames: list
                contains previous and current list of Detections
            fps: float or int
                actual frame per second. If frame_interval != 1, it doesn't equal the original value,
        Returns:
            feature_vec_combined: list of Tensor
                [geo_prev, depth_prev, geo_curr, depth_curr], dimension:
                [6, 13*13, 6, 13*13] -> [1, 350]
                size of depth_feature_vector, i.e. 13, might be adjusted.
        """
        dets_prev, dets_curr = dets_neighbor_frames
        fps = tf.reshape(tf.constant([fps], dtype=tf.float32), (1, -1))
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
                fps,
                tf.reshape(det.get_geo_clues(), (1, -1)),
                tf.reshape(det.get_depth_feature(), (1, -1))
            ], axis=1)
            feature_vec_curr.append(fused_feature)

        feature_vec_combined = [tf.concat([vec_prev, vec_curr], axis=1) for vec_prev, vec_curr
                                in zip(feature_vec_prev, feature_vec_curr)]
        return feature_vec_combined

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
                fx / (r - l), fy / (t - b),
                (l - pp_x) / fx, (r - pp_x) / fx,
                (t - pp_y) / fx, (b - pp_y) / fx
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

    def resize_depth_map_pair(self, depth_maps, dets_neighbor_frames):
        """From depth map patches get feature vector
        Args:
            depth_maps: list
                two depth maps, each in shape (1, 192, 640, 1)
            dets_neighbor_frames: list
                Detections for two neighbor frames, must make sure:
                - the IDs between boxes of two frames are one-to-one associated
                - the list is ordered by ID, therefore it's ok to directly unpack
        Returns:
            list of feature vectors
        """

        def resize_map2vec_per_bbox(depth_map, bbox_ltrb):
            assert len(depth_map.shape) in (3, 4) and depth_map.shape[-1] == 1, \
                "depth_map should have shape (1,192,640,1) or (192, 640, 1), got {}".format(depth_map.shape)
            if len(depth_map.shape) == 3:
                depth_map = np.expand_dims(depth_map, 0)
            l, t, r, b = bbox_ltrb
            feature_map = depth_map[:, t:b, l:r, :]
            feature_map = tf.image.resize(feature_map, (27, 27))
            feature_map = MaxPool2D((3, 3), strides=2)(feature_map)
            feature_vec = tf.reshape(feature_map, (-1))
            return feature_vec

        def offset_and_rescale_bbox(det, offset_y=168, image_scaling=2):
            skip, bbox = Detection.offset_bbox(det.get_ltwh(),
                                               offset_y=offset_y, boundaries=(0, 192 * image_scaling))
            bbox = np.array(bbox) / 2
            bbox[2:] += bbox[:2]
            ltrb = bbox.astype(np.int32).tolist()
            return ltrb

        def resize_map2vec_all_bboxes(dets, depth_map):
            for det in dets:
                bbox_ltrb = offset_and_rescale_bbox(det)
                det.receive_depth_feature(
                    resize_map2vec_per_bbox(depth_map, bbox_ltrb)
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

    def disp_to_depth(self, disp, flatten=False):
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
            depth_input.append(image_cropped)
        depth_input = tf.stack(depth_input)
        depth_input = tf.image.resize(depth_input, self.feed_size)
        return depth_input, offset_y


def main(_):
    opt = get_options()
    trainer_velo = TrainerVelo(opt)
    trainer_velo.train()


if __name__ == '__main__':
    from options_velomlp import get_options
    from absl import app
    app.run(main)
