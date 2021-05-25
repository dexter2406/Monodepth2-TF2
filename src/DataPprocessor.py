import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from utils import *


class DataProcessor(object):
    def __init__(self, frame_idx, feed_size, intrinsics=None, num_scales=4,
                 disable_gt=False, dataset=None):
        self.dataset_name = dataset.name
        self.disable_gt = disable_gt
        self.num_scales = num_scales     # for evaluation, 1
        self.height, self.width = feed_size
        self.frame_idx = frame_idx
        self.batch_size = -1
        self.K = intrinsics
        self.brightness = 0.3
        self.sature_contrast = 0.2
        self.hue = 0.15
        self.asp_ratio = self.width / self.height
        self.usual_asp_ratios : tuple = None
        self.asp_ratio_probs = None
        self.init_asp_ratio_probs()
        self.raw_size = dataset.full_res_shape

    def init_asp_ratio_probs(self):
        # https://en.wikipedia.org/wiki/Aspect_ratio_(image)
        self.usual_asp_ratios = (1.37, 1.43, 14 / 9, 1.6, 1.78, 16 / 9, 1.85, 2.0, 2.165, 2.39, 2.59, 8 / 3, 2.76, 640 / 192)
        num = len(self.usual_asp_ratios)
        self.asp_ratio_probs = np.ones(num)
        for i in range(num):
            if self.usual_asp_ratios[i] == 1.78:
                self.asp_ratio_probs[i] = 2
            elif self.usual_asp_ratios[i] == 640 / 192:
                self.asp_ratio_probs[i] = 3
        self.asp_ratio_probs = self.asp_ratio_probs / np.sum(self.asp_ratio_probs)

    def get_intrinsics(self, K):
        self.K = K

    def disable_groundtruth(self):
        self.disable_gt = True

    def prepare_batch(self, batch, is_train, random_crop=False):
        """Prepare batch data

        Outputs: raw and augmented data, stored in dictionary
        Dictionary keys:
        ("color", <frame_id>, <scale>)          for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K", scale) or ("inv_K", scale)        for camera intrinsics,
        ('val_mask', 0, 0)                      to mask out moving object for pose input and reprojection loss
        ('far_bbox', 0, 0)                      to put constraint on disp/depth map for small object
        ------- Examples -------
        tgt_image = inputs['color', 0, 0]
        src_image_stack_aug = inputs['color_aug', 1:, :]
        tgt_image_pyramid = inputs['color, 0, :]
        intrinsics_scale_0 = inputs[('K', 0)]
        """
        batch_imgs, depth_gt, ext_bbox = self.decompose_inputs(batch)
        do_color_aug = is_train and np.random.random() > 0.5
        # due to tf.function, better explicitly pass the parameter
        inputs = self.process_imgs(batch_imgs, random_crop, do_color_aug, depth_gt=depth_gt)
        input_Ks = self.make_K_pyramid(batch_imgs.shape[0])
        inputs.update(self.process_ext_bboxes(ext_bbox))
        return inputs, input_Ks

    def process_ext_bboxes(self, ext_bbox):
        """Handle the external bboxes
        -> if not available, store as None
        -> if available but the box doesn't exit, make sure it's [0, 0, 0, 0]*6
        """
        inputs = {}
        if ext_bbox is not None:
            inputs[('val_mask', 0, 0)] = self.create_validity_mask(ext_bbox)
            inputs.update(self.collect_min_box(ext_bbox))
        else:
            inputs[('val_mask', 0, 0)] = None
            for f_i in self.frame_idx:
                inputs[('far_bbox', f_i, 0)] = None
        return inputs

    def collect_min_box(self, ext_bbox):
        """collect box wiht minimum size to put constrain on far object
        The identities of the boxes are not necessarily be the same, unlike validity_mask
        Args:
            ext_bbox: Tensor, (B,6,4)
                box convention: left, top, right, bottom
                [B,:3,:], [B,3:] are max and min bboxes respectively, only the [B,3:] used here,
                representing boxes in prev, curr and next frame, respectively.
        """

        inputs = {('far_bbox', 0, 0):   tf.expand_dims(ext_bbox[:, :2], 1),
                  ('far_bbox', -1, 0):  tf.expand_dims(ext_bbox[:, 2:4], 1),
                  ('far_bbox', 1, 0):   tf.expand_dims(ext_bbox[:, 4:], 1)}
        return inputs

    def create_validity_mask(self, ext_bbox):
        """Create validity mask using bboxes in the triplet
        -> Rough judge if they are the same identity by IoU
        -> Select bbox(es), merge and create mask in the background with size of self.inp_size
        Note: the boxes doesn't necessarily represent the same identity, thus we have to do post processing
            to see how to make use of them
        Args:
            ext_bbox: Tensor, (B,6,4)
                box convention: left, top, right, bottom
                [B,:3,:], [B,3:] are max and min bboxes respectively, only the [B,:3] used here,
                representing boxes in prev, curr and next frame, respectively.
        Returns:
            inputs: dict with key ('val_mask',0,0)
                 shape (B, *inp_size)
        """
        boxes = ext_bbox[:, :3].numpy()
        boxes = self.dilate_boxes(boxes)
        box_merged = self.get_merged_box(boxes)
        val_mask = self.place_box_on_background(box_merged)
        if len(val_mask.shape) == 3 and val_mask.shape[-1] != 1:
            val_mask = tf.expand_dims(val_mask, -1)
        return val_mask

    def dilate_boxes(self, boxes):
        """dilate boxes to ensure coverage
        """
        offset_ratio = [0.15, 0.4]     # up/down. left/right direction
        dilate_factor = [1.1, 0.7]     # in H, W direction, respectively
        boxes = boxes.astype(np.float32)

        heights = (boxes[..., 3] - boxes[..., 1]) * dilate_factor[0]
        boxes[..., 1] -= heights * offset_ratio[0]
        boxes[..., 3] += heights * (1-offset_ratio[0])

        widths = (boxes[..., 2] - boxes[..., 0]) * dilate_factor[1]
        boxes[..., 0] -= widths * offset_ratio[1]
        boxes[..., 2] += widths * (1-offset_ratio[1])
        return boxes.astype(np.int32)

    def place_box_on_background(self, box):
        batch_size = box.shape[0]
        val_mask = np.ones(shape=(batch_size, self.height, self.width), dtype=np.float32)
        for b in range(batch_size):
            val_mask[b, box[b, 1]: box[b, 3], box[b, 0]: box[b, 2]] = 0.
        return val_mask

    def get_merged_box(self, box_batch):
        """Merge intersected boxes, if no intersection, choose the largest one
        Args:
            box_batch: Tensor, shape (B,3,4)
        """
        batch_size, frame_num = box_batch.shape[:2]
        # ------------------------
        # Batch IoU, pair-wise comparison
        # each (B,1) for prev-curr, curr-next or next-prev
        # ------------------------
        IOUs = []
        for f_i in range(frame_num):
            boxes1, boxes2 = box_batch[:, f_i], box_batch[:, (f_i+1) % 3]   # compare 01, 12, 20
            iou_pair = self.batch_iou(boxes1, boxes2)                   # (B,1)
            IOUs.append(iou_pair)                                       # [(B,1), (B,1), (B,1)]
        # ------------------------
        # One batch has 3 boxes, thus shape (B,3,4)
        # Merge 3 boxes in to one, then stack back to batch_size, namely (B,4)
        # We have to see run over each triplets, so this cannot be batch-processed
        # ------------------------
        merged_boxes = []
        for b in range(batch_size):
            boxes = box_batch[b]
            box_merged = None
            # -------------------------
            # list of (B,3*1), meaning each box for prev, curr and next frame
            # -------------------------
            box_areas = []
            for f_i in range(frame_num):
                box1, box2 = boxes[f_i], boxes[(f_i + 1) % 3]
                iou = IOUs[f_i][b]
                box_w = boxes[f_i, 2] - boxes[f_i, 0]
                box_h = boxes[f_i, 3] - boxes[f_i, 1]
                box_areas.append(box_w*box_h)
                if iou > 0.2:
                    box_merged_tmp = merge_boxes(box1, box2)
                    if box_merged is None:
                        box_merged = box_merged_tmp
                    else:
                        box_merged = merge_boxes(box_merged, box_merged_tmp)    # (4,)
            if box_merged is None:
                box_merged = boxes[tf.argmax(box_areas)]
                box_merged = tf.squeeze(tf.constant(box_merged))
            merged_boxes.append(box_merged)

        merged_boxes = np.stack(merged_boxes)  # (B,4)
        return merged_boxes

    def batch_iou(self, a, b, epsilon=1e-5):
        """ Given two arrays `a` and `b` where each row contains a bounding
            box defined as a list of four numbers:
                [x1,y1,x2,y2]
            where:
                x1,y1 represent the upper left corner
                x2,y2 represent the lower right corner
            It returns the Intersect of Union scores for each corresponding
            pair of boxes.

        Args:
            a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
            b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
            epsilon:    (float) Small value to prevent division by zero

        Returns:
            (numpy array) The Intersect of Union scores for each pair of bounding
            boxes.
        """
        # COORDINATES OF THE INTERSECTION BOXES
        x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
        y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
        x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
        y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

        # AREAS OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)

        # handle case where there is NO overlap
        width[width < 0] = 0
        height[height < 0] = 0

        area_overlap = width * height

        # COMBINED AREAS
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined + epsilon)
        return iou

    def decompose_inputs(self, batch):
        depth_gt = None
        ext_bbox = None
        if isinstance(batch, tuple):
            batch_imgs = batch[0]
            for item in batch[1:]:
                if item.shape[-1] == 4:
                    ext_bbox = item  # shape of (B,3*2,4)
                else:
                    depth_gt = tf.expand_dims(item, 3)
        else:
            batch_imgs = batch
        if depth_gt is not None:
            assert 1.3 < depth_gt.shape[2] / depth_gt.shape[1] < 3.5, \
                "judge from aspect ratio, deoth_gt shape might be wrong: {}".format(depth_gt.shape)
        return batch_imgs, depth_gt, ext_bbox

    def prepare_batch_val(self, batch, random_crop=False):
        """duplicate of prepare_batch(), no @tf.function decorator
        For validation OR evaluation
        """
        batch_imgs, depth_gt, ext_bbox = self.decompose_inputs(batch)
        inputs = self.process_imgs(batch_imgs, random_crop=random_crop, do_color_aug=False, depth_gt=depth_gt)
        input_Ks = self.make_K_pyramid(batch_imgs.shape[0])
        return inputs, input_Ks

    @tf.function
    def process_imgs(self, batch_imgs, random_crop=False, do_color_aug=False, depth_gt=None):
        """Preprocess images
        Args:
            batch_imgs, Tensor, shape of (B,H,W,9)
                -> tgt_batch has 3, src_batch has 2*3
        """
        inputs = {}
        if depth_gt is not None:
            inputs[('depth_gt', 0, 0)] = depth_gt
        tgt_batch, src_batch = batch_imgs[..., :3], batch_imgs[..., 3:]

        # in training / eval_depth / eval_pose
        # frame_idx == [0,-1,1] / [0] / [0,1] respectively
        inputs[('color', self.frame_idx[0], -1)] = tgt_batch
        if len(self.frame_idx) > 1:
            for i, f_i in enumerate(self.frame_idx[1:]):
                inputs[('color', f_i, -1)] = src_batch[..., i*3: (i+1)*3]
        inputs = self.preprocess(inputs, do_color_aug, random_crop)

        return inputs

    def generate_aug_params(self, do_color_aug=False):
        """Generate augmentation probability during training
        For velocity_challenge, augmentation is less, since the scene has bad light condition
        """
        if do_color_aug:
            brightness = np.random.uniform(-self.brightness, self.brightness)
            hue = np.random.uniform(-self.hue, self.hue)
            saturation, contrast = np.random.uniform(1 - self.sature_contrast, 1 + self.sature_contrast, size=2)
            do_flip, do_blur = np.random.uniform(0, 1, size=2)
            aug_params = {
                'brightness': brightness,
                'saturation': saturation,
                'contrast': contrast,
                'hue': hue,
                'do_flip': do_flip,
                'do_blur': do_blur
            }
            if 'velo' in self.dataset_name:
                for k in aug_params:
                    aug_params[k] /= 2
            return aug_params

        else:
            return None

    def color_aug(self, image, aug_params):
        """Apply augmentation
        Same aug for each batch (and scales), Note that all images input to the pose network receive the
        same augmentation.
        """
        if aug_params is not None:
            image = tf.image.adjust_brightness(image, aug_params['brightness'])
            image = tf.image.adjust_saturation(image, aug_params['saturation'])
            image = tf.image.adjust_contrast(image, aug_params['contrast'])
            image = tf.image.adjust_hue(image, aug_params['hue'])
            if aug_params['do_flip'] > 0.5:
                image = tf.image.flip_left_right(image)     # also applied to gt
            if aug_params['do_blur'] > 0.5:
                image = tfa.image.gaussian_filter2d(image, (3, 3), 1.0, 'REFLECT', 0)
        return image

    def preprocess(self, inputs, do_color_aug=False, random_crop=False):
        """Make pyramids and augmentations
        - pyramid: use the raw (scale=-=1) to produce scale==[0:4] images
        - augment: correspond to the pyramid source
        """
        # generate one set of random aug factor for one batch, making that same pair has same aug effects
        aug_params = self.generate_aug_params(do_color_aug)
        asp_ratio = self.gen_asp_ratio(random_crop)
        shape_s0 = (None, None)     # H,W

        for k in list(inputs):
            if ('depth_gt', 0, 0) == k:
                inputs[k] = self.random_width_crop(inputs[k], asp_ratio)
                if aug_params is not None and aug_params['do_flip'] > 0.5:
                    inputs[k] = tf.image.flip_left_right(inputs[k])
                assert 1.3 < inputs[k].shape[2] / inputs[k].shape[1] < 3.5, \
                    "judging from aspect ratio, deoth_gt shape might be wrong, got {}".format(inputs[k].shape)
            else:
                img_type, f_i, scale = k
                if img_type == 'color':
                    for scale in range(self.num_scales):
                        image = inputs[(img_type, f_i, scale - 1)]
                        if scale == 0:
                            # random crop to (H, H*A) to simulate different resizing effect
                            image = self.random_width_crop(image, asp_ratio)
                            shape_s0 = image.shape[1:3]
                        resize_to = (shape_s0[0] // (2 ** scale), shape_s0[1] // (2 ** scale))
                        image = tf.image.resize(image, resize_to, antialias=True)
                        inputs[(img_type, f_i, scale)] = image
                        inputs[(img_type + '_aug', f_i, scale)] = self.color_aug(image, aug_params)
        return inputs

    @tf.function
    def make_K_pyramid(self, batch_size):
        """genearing intrinsics pyramid
        Args:
            batch_size: for exp purpose, validation set has smaller batch_size, which needs to be
                explicitly passed when using @tf.function
        Returns:
            input_Ks: dict
                a pyramid of homogenous intrinsics and its inverse, each has shape [B,4,4]
        """
        def get_K_scaling():
            # For VelocityChallenge, aspect ratio 1280/720 is center-cropped to fit 640/192
            # Therefore the scaling_fx is directly self.width,
            # but scaling_fy is obtained by width ratio, instead of directly using self.height
            raw_aspr = self.raw_size[1] / self.raw_size[0]
            feed_aspr = self.width / self.height
            scalings = [self.width, self.height]
            if abs(raw_aspr - feed_aspr) > 0.02:
                scalings[1] = self.raw_size[0] / (self.raw_size[1] / self.width)
            return scalings

        input_Ks = {}
        scalings = get_K_scaling()
        # Same intrinsics for all data
        for scale in range(self.num_scales):
            K_tmp = self.K.copy()
            K_tmp[0, :] *= scalings[0] // (2 ** scale)
            K_tmp[1, :] *= scalings[1] // (2 ** scale)

            inv_K = np.linalg.pinv(K_tmp)

            K_tmp = tf.reshape(tf.tile(K_tmp, [batch_size, 1]), (batch_size, 4, 4))
            inv_K = tf.reshape(tf.tile(inv_K, [batch_size, 1]), (batch_size, 4, 4))

            input_Ks[("K", scale)] = K_tmp
            input_Ks[("inv_K", scale)] = inv_K
        # print(input_Ks[("K", 0)][0])
        return input_Ks

    def make_K_pyramid_normal(self, external_K):
        """genearing intrinsics pyramid
        Args:
            K: partial intrinsics, shape best to be [B,3,3], but [B,4,4], [4,4] or [B,4,4] are also OK
        Returns:
            input_Ks: dict
                a pyramid of homogenous intrinsics and its inverse, each has shape [B,4,4]
        """
        # NOT IN USE
        input_Ks = {}
        # Same intrinsics for all data
        for scale in range(self.num_scales):
            K_tmp = external_K.numpy()
            K_tmp[:, 0, :] *= self.width // (2 ** scale)
            K_tmp[:, 1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K_tmp)

            input_Ks[("K", scale)] = K_tmp
            input_Ks[("inv_K", scale)] = inv_K
        # assert_valid_hom_intrinsics(input_Ks[("K", 0)])
        return input_Ks

    def delete_raw_images(self, inputs, scale_to_del=-1):
        for idx in self.frame_idx:
            del inputs[('color', self.frame_idx[idx], scale_to_del)]

    def gen_asp_ratio(self, random_crop=False):
        """generate aspect ratio for one batch"""
        # Not In Use
        if random_crop:
            asp_ratio = np.random.choice(self.usual_asp_ratios, 1, p=self.asp_ratio_probs).astype(np.float32)[0]
        else:
            asp_ratio = None
        return asp_ratio

    def random_width_crop(self, images, asp_ratio=None):
        """Symmetric width crop
        H preserved; W = H * asp_ratio
        """
        # Not In Use
        if asp_ratio:
            h, w = images.shape[1:3]
            w_goal = min(w,h * asp_ratio)
            w_start = 0 + (w - w_goal) // 2
            w_end = min(w, w_start + w_goal)
            images = images[:, :, int(w_start): int(w_end), :]
        return images
