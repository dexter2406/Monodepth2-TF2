from src.DepthEstimationModel import DepthEstimationModel
from src.depth_estimation import run_estimation
import cv2 as cv
from collections import defaultdict
import time


def init_depth_estimator():
    weights_dir: str = './models'
    encoder_name: str = 'resnet18'
    decoder_name = 'depth_decoder'
    single_mode = True
    depth_estimator = DepthEstimationModel(weights_dir=weights_dir,
                                           encoder_name=encoder_name, decoder_name=decoder_name,
                                           esimate_func=run_estimation, single_mode=True)
    return depth_estimator


def test():
    depth_estimator = init_depth_estimator()

    vid_path = [r"D:\MA\motion_ds\SeattleStreet_1.mp4",
                r"D:\MA\Struct2Depth\KITTI_odom_02\seq2.avi"]
    cap = cv.VideoCapture(vid_path[1])
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    orig_w, orig_h = int(cap.get(3)), int(cap.get(4))
    # depth_estimator.original_size = (orig_w, orig_h)
    # writer = cv.VideoWriter('depth_monodepth2.avi', fourcc, 12, (orig_w, orig_h))

    timers = defaultdict(lambda : 1e-4)
    while cap.isOpened():
        ret, image = cap.read()
        if type(image) is None:
            exit("wrong video")
        t0 = time.time()
        colormapped_im, _ = depth_estimator.esimate_func(image, depth_estimator)
        cv.imshow('', colormapped_im)
        # writer.write(colormapped_im)
        if cv.waitKey(1) & 0xFF==27:
            break

if __name__ == '__main__':
    test()