import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, MaxPool2D
from tensorflow.keras.models import Model
import numpy as np
from src.detection import Detection
from utils import video_to_numpy
import json


class VelocityMLP(Model):
    def __init__(self):
        super(VelocityMLP, self).__init__()
        prefix = 'velo_dense'
        self.fc_0 = Dense(256, activation='relu', use_bias=True, name=prefix+'_0')
        self.fc_1 = Dense(128, activation='relu', use_bias=True, name=prefix+'_1')
        self.fc_2 = Dense(64, activation='relu', use_bias=True, name=prefix+'_2')
        self.fc_3 = Dense(4, use_bias=True, name=prefix+'_3')

    def call(self, inputs, training=None, mask=None):
        out = self.fc_0(inputs)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)
        return out


def test():
    velocity_mlp = VelocityMLP()
    video_idx = 12
    frame_interval = 2
    fps = 20
    cam_params = {
        'focal_x': 714.2, 'focal_y': 710.4,
        'pp_x': 675.6, 'pp_y': 376.3,
        'height': 1.8}
    actual_fps = fps / frame_interval

    depth_maps, bboxes_per_frame = get_depth_maps_and_detections(video_idx, frame_interval)
    depth_map_pair, bboxes_pair, gt_motion, valid_num = get_neighbor_pair(depth_maps, bboxes_per_frame, frame_interval)
    depth_map_pairs, bboxes_pairs, gt_motions, valid_nums = create_batch([depth_map_pair, bboxes_pair, gt_motion, valid_num], batch_size=5)

    loss_per_obj = []
    print(depth_map_pairs.shape, bboxes_pairs.shape, gt_motions.shape, valid_nums.shape)
    for depth_map_pair, bboxes_pair, gt_motion, valid_num in zip(depth_map_pairs, bboxes_pairs, gt_motions, valid_nums):
        bboxes_pair_valid = [bboxes[:valid_num] for bboxes in bboxes_pair]
        feature_vector_per_obj = get_aggreated_features(depth_map_pair, bboxes_pair_valid, cam_params, fps=actual_fps)
        motion_data_per_obj = []
        for feat_vec in feature_vector_per_obj:
            motion_data_per_obj.append(
                np.squeeze(velocity_mlp(feat_vec)).tolist()
            )
        motion_preds = tf.stack(motion_data_per_obj)
        gt_motion = gt_motion[0][:valid_num]
        # print("valid_num", valid_num)
        # print("gt", gt_motion.shape, '\n', gt_motion)
        # print("pred\n", motion_preds.shape, '\n', motion_preds)
        loss_batch = gt_motion - motion_preds
        for i in range(valid_num):
            loss_per_obj.append(loss_batch[i])
    loss_batch = tf.stack(loss_per_obj)
    print("loss_all:\n", loss_batch)
    print('loss_mean', tf.reduce_mean(loss_batch))


def create_batch(data_list, batch_size):
    for i in range(len(data_list)):
        data_list[i] = tf.constant([data_list[i]] * batch_size)
    return data_list


def get_depth_maps_and_detections(vid_idx, frame_interval):
    # use RGB image to substitute depth map for experiments
    image_stack = video_to_numpy(r'F:\Dataset\VelocityChallenge\benchmark_velocity_train\clips_in_video\{:03}.avi'.
                                 format(vid_idx))
    image_stack = image_stack[-frame_interval-1:]
    depth_map_substitutes = [image[..., 0:1] for image in image_stack]
    # depth_map_substitutes = [tf.image.resize(image, (192, 640)).numpy() for image in depth_map_substitutes]

    # load bbox_ltwh for corresponding frames
    with open(r'F:\Dataset\VelocityChallenge\benchmark_velocity_train\gt_bboxes\{:03}.json'.format(vid_idx)) as jf:
        bboxes_per_frame = json.load(jf)
    bboxes_per_frame = bboxes_per_frame[-frame_interval-1:]

    return depth_map_substitutes, bboxes_per_frame


def get_aggreated_features(depth_map_pair, bboxes_pair, cam_params, fps):
    """Generate feature vector from previous and current depth map and geometry parameter
    It only applies to a simple case:
    -> All current bboxes have corresponding bbox in the previous frame (-1 or -2).
    This is for VelocityChallenge dataset, where tracked bboxes are pre-computed

    Args:
        depth_maps: list
            list of depth / disp map, in temporal order
        dets_latest_frames: list of list
            list of Detection, in temporal order. Empty list means nothing detected
        cam_params: dict
            camera parameters like principle point, focal length
        frame_interval: int
            More interval can tolerate the scale-shift perturbation in depth estimation
        fps: float or int
            frame-per-second.
    """
    dets_pair = []
    for bboxes in bboxes_pair:
        dets_pair.append([Detection(ltwh=bbox) for bbox in bboxes])
    dets_pair = associate_IDs(dets_pair)
    dets_pair = calc_geometry_clues(dets_pair, cam_params)
    dets_pair = resize_depth_map_pair(depth_map_pair, dets_pair)
    feature_vec = aggregate_geo_clues_and_depth_feature(dets_pair, fps)
    return feature_vec


def get_neighbor_pair(depth_maps, bboxes_per_frame, frame_interval):
    """Select depth and detection pairs from two target frames"""
    bboxes_pair = [bboxes_per_frame[-frame_interval - 1],
                   bboxes_per_frame[-1]]
    valid_num = len(bboxes_pair[0])
    bboxes_pair = complete_4_items(bboxes_pair)

    depth_maps = [depth_maps[-frame_interval-1],
                  depth_maps[-1]]
    gt_motion_seed = [[1., 1., 2., 2.]] * valid_num
    gt_motion = complete_4_items([gt_motion_seed])

    return depth_maps, bboxes_pair, gt_motion, valid_num


def associate_IDs(dets_latest_frames):
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


def calc_geometry_clues(dets_neighbor_frames, cam_params):
    """assign geometry clues in each Detection"""

    def calc_geo_clues(det, cam_params):
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
        geo_clues = calc_geo_clues(det, cam_params)
        det.receive_geo_clues(
            np.array(geo_clues, dtype=np.float32)
        )

    for det in dets_neighbor_frames[1]:
        geo_clues = calc_geo_clues(det, cam_params)
        det.receive_geo_clues(
            np.array(geo_clues, dtype=np.float32)
        )

    return dets_neighbor_frames


def resize_depth_map_pair(depth_maps, dets_neighbor_frames):
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


def aggregate_geo_clues_and_depth_feature(dets_neighbor_frames, fps):
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


def offset_and_rescale_bbox(det, offset_y=168, image_scaling=2):
    skip, bbox = Detection.offset_bbox(det.get_ltwh(),
                                       offset_y=offset_y, boundaries=(0, 192*image_scaling))
    bbox = np.array(bbox) / 2
    bbox[2:] += bbox[:2]
    ltrb = bbox.astype(np.int32).tolist()
    return ltrb





def complete_4_items(data_lists):
    """Make each list in data_list have 4 elements.
    If not enough, duplicate; if exceeds, crop the first 4
    """
    num_lists = len(data_lists)
    for i in range(num_lists):
        num_elem = len(data_lists[i])
        if num_elem == 1:
            data_lists[i] = data_lists[i] * 4
        elif num_elem == 2:
            data_lists[i] = data_lists[i] * 2
        elif num_elem == 3:
            data_lists[i] = data_lists[i].append(data_lists[i][0])
        elif num_elem > 4:
            data_lists[i] = data_lists[i][:4]
    return data_lists


if __name__ == '__main__':
    test()
    # test_maxpool()

