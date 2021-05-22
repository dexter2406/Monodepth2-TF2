import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class DataProcessor(object):
    def __init__(self, frame_idx, inp_size, intrinsics=None, num_scales=4, disable_gt=False, dataset_name=None):
        self.dataset_name = dataset_name
        self.disable_gt = disable_gt
        self.num_scales = num_scales     # for evaluation, 1
        self.height, self.width = inp_size
        self.frame_idx = frame_idx
        self.batch_size = -1
        self.K = intrinsics
        self.brightness = 0.25
        self.sature_contrast = 0.2
        self.hue = 0.1
        self.asp_ratio = self.width / self.height
        self.usual_asp_ratios : tuple = None
        self.asp_ratio_probs = None
        self.init_asp_ratio_probs()

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
        """Apply augmentation
        tgt_batch: ndarray
            - [<Batch>, height, width, 3], for original monodepth2 (192, 640)
        src_batch: ndarray:
            - [<Batch>, height, width, 6]

        Outputs: raw and augmented data, stored in dictionary
        Dictionary keys:
        ("color", <frame_id>, <scale>)          for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K", scale) or ("inv_K", scale)        for camera intrinsics,
        ------- Examples -------
        tgt_image = inputs['color', 0, 0]
        src_image_stack_aug = inputs['color_aug', 1:, :]
        tgt_image_pyramid = inputs['color, 0, :]
        intrinsics_scale_0 = inputs[('K', 0)]
        """
        batch_imgs, depth_gt, ext_K = self.decompose_inputs(batch)
        do_color_aug = is_train and np.random.random() > 0.5
        # due to tf.function, better explicitly pass the parameter
        inputs = self.process_imgs(batch_imgs, random_crop, do_color_aug, depth_gt=depth_gt)

        # todo: how to deal with tf.function
        if ext_K is None:
            input_Ks = self.make_K_pyramid_fast(batch_imgs.shape[0])
        else:
            input_Ks = self.make_K_pyramid_normal(ext_K)
        return inputs, input_Ks

    def decompose_inputs(self, batch):
        depth_gt = None
        ext_K = None
        if isinstance(batch, tuple):
            batch_imgs = batch[0]
            for item in batch[1:]:
                if item.shape[-1] == 4:
                    ext_K = item    # external intrinsics NOT IN USE for now
                    assert ext_K.shape[1] == ext_K.shape[2], \
                        "intrinsics must have dimension (B,4,4), got{}".format(self.K.shape)
                else:
                    depth_gt = tf.expand_dims(item, 3)
        else:
            batch_imgs = batch
        if depth_gt is not None:
            assert 1.3 < depth_gt.shape[2] / depth_gt.shape[1] < 3.5, \
                "judge from aspect ratio, deoth_gt shape might be wrong: {}".format(depth_gt.shape)
        return batch_imgs, depth_gt, ext_K

    def prepare_batch_val(self, batch, random_crop=False):
        """duplicate of prepare_batch(), no @tf.function decorator
        For validation OR evaluation
        """
        batch_imgs, depth_gt, ext_K = self.decompose_inputs(batch)
        inputs = self.process_imgs(batch_imgs, random_crop=random_crop, do_color_aug=False, depth_gt=depth_gt)
        input_Ks = self.make_K_pyramid_fast(batch_imgs.shape[0])
        return inputs, input_Ks

    @tf.function
    def process_imgs(self, batch_imgs, random_crop=False, do_color_aug=False, depth_gt=None):
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
    def make_K_pyramid_fast(self, batch_size):
        """genearing intrinsics pyramid
        Args:
            batch_size: for exp purpose, validation set has smaller batch_size, which needs to be
                explicitly passed when using @tf.function
        Returns:
            input_Ks: dict
                a pyramid of homogenous intrinsics and its inverse, each has shape [B,4,4]
        """
        input_Ks = {}
        # Same intrinsics for all data
        for scale in range(self.num_scales):
            K_tmp = self.K.copy()
            K_tmp[0, :] *= self.width // (2 ** scale)
            K_tmp[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K_tmp)

            K_tmp = tf.reshape(tf.tile(K_tmp, [batch_size, 1]), (batch_size, 4, 4))
            inv_K = tf.reshape(tf.tile(inv_K, [batch_size, 1]), (batch_size, 4, 4))

            input_Ks[("K", scale)] = K_tmp
            input_Ks[("inv_K", scale)] = inv_K
        # assert_valid_hom_intrinsics(input_Ks[("K", 0)])
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
