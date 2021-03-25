# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import pickle
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


def check_weights(weights_data):
    for weights in weights_data:
        print("length: ", len(weights))
        for i in range(2):
            print(type(weights[0]), weights[0].shape)
            print(type(weights[1]), weights[1].shape)


def extract_weights(encoder):

    """
    encoder._modules['encoder']
    # Conv1
    .conv1.weight
    # BN1
    .bn1.weight
    .bn1.bias
    .bn1.running_mean
    .bn1.running_var
    """
    """
    # ResNetBlock, 如 encoder._modules['encoder'].layer4._modules['1'].conv2.weight
    .layer<1~4>._modules['0或1']
    # # Conv
    ..conv<1或2>.weight
    # # BN: ..bn<1或2> + 4个变量
    ..bn<>.weight    
    ..bn<>.bias
    ..bn<>.running_mean
    ..bn<>.running_var
    """

    enc_weights = []
    enc_modules = encoder._modules['encoder']
    enc_weights.append(enc_modules.conv1.weight)
    enc_weights.append(enc_modules.bn1.weight)
    enc_weights.append(enc_modules.bn1.bias)
    enc_weights.append(enc_modules.bn1.running_mean)
    enc_weights.append(enc_modules.bn1.running_var)
    enc_resblock_weights = []
    # layer1
    for i in range(2):
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].conv1.weight)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.weight)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.bias)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.running_mean)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.running_var)

        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].conv2.weight)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.weight)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.bias)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.running_mean)
        enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.running_var)

    # layer2
    for i in range(2):
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].conv1.weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.bias)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.running_mean)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.running_var)

        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].conv2.weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.bias)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.running_mean)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.running_var)

        if i == 0:
            enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['0'].weight)
            enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].weight)
            enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].bias)
            enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_mean)
            enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_var)

    # layer3
    for i in range(2):
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].conv1.weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.bias)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.running_mean)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.running_var)

        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].conv2.weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.bias)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.running_mean)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.running_var)

        if i == 0:
            enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['0'].weight)
            enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].weight)
            enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].bias)
            enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_mean)
            enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_var)

    # layer4
    for i in range(2):
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].conv1.weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.bias)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.running_mean)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.running_var)

        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].conv2.weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.bias)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.running_mean)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.running_var)

        if i == 0:
            enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['0'].weight)
            enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].weight)
            enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].bias)
            enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_mean)
            enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_var)

    enc_weights.extend(enc_resblock_weights)
    print("transform weights to numpy, transpose, and save ... ")
    outputs = []
    for i, w in enumerate(enc_weights):
        w = to_numpy(w)
        print("before:", w.shape)
        if len(w.shape) == 4:
            w = np.transpose(w, [2, 3, 1, 0])
        outputs.append(w)
        print("\tSaving ", w.shape, "==============")
    with open('models/enc_weights_enumerate.pkl', 'wb') as df:
        pickle.dump(outputs, df)

    exit("Done saving")

""" 
Extract weights LAYERWISE
encoder._modules['encoder']
# Conv1
.conv1.weight
# BN1
.bn1.weight
.bn1.bias
.bn1.running_mean
.bn1.running_var
"""
"""
# ResNetBlock, 如 encoder._modules['encoder'].layer4._modules['1'].conv2.weight
.layer<1~4>._modules['0或1']
# # Conv
..conv<1或2>.weight
# # BN: ..bn<1或2> + 4个变量
..bn<>.weight    
..bn<>.bias
..bn<>.running_mean
..bn<>.running_var
"""
"""
enc_weights = []
enc_modules = encoder._modules['encoder']
enc_weights.append([enc_modules.conv1.weight])
bn_tmp = [enc_modules.bn1.weight,
          enc_modules.bn1.bias,
          enc_modules.bn1.running_mean,
          enc_modules.bn1.running_var]
enc_weights.append(bn_tmp)
# enc_weights.append(enc_modules.conv1.weight)
# enc_weights.append(enc_modules.bn1.bias)
# enc_weights.append(enc_modules.bn1.running_mean)
# enc_weights.append(enc_modules.bn1.running_var)
enc_resblock_weights = []
# layer1
for i in range(2):
    enc_resblock_weights.append([enc_modules.layer1._modules['%d' % i].conv1.weight])
    bn_tmp = [enc_modules.layer1._modules['%d' % i].bn1.weight,
              enc_modules.layer1._modules['%d' % i].bn1.bias,
              enc_modules.layer1._modules['%d' % i].bn1.running_mean,
              enc_modules.layer1._modules['%d' % i].bn1.running_var]
    enc_resblock_weights.append(bn_tmp)

    enc_resblock_weights.append([enc_modules.layer1._modules['%d' % i].conv2.weight])
    bn_tmp = [enc_modules.layer1._modules['%d' % i].bn2.weight,
              enc_modules.layer1._modules['%d' % i].bn2.bias,
              enc_modules.layer1._modules['%d' % i].bn2.running_mean,
              enc_modules.layer1._modules['%d' % i].bn2.running_var]
    enc_resblock_weights.append(bn_tmp)

# layer2
for i in range(2):
    enc_resblock_weights.append([enc_modules.layer2._modules['%d' % i].conv1.weight])
    bn_tmp = [enc_modules.layer2._modules['%d' % i].bn1.weight,
              enc_modules.layer2._modules['%d' % i].bn1.bias,
              enc_modules.layer2._modules['%d' % i].bn1.running_mean,
              enc_modules.layer2._modules['%d' % i].bn1.running_var]
    enc_resblock_weights.append(bn_tmp)

    enc_resblock_weights.append([enc_modules.layer2._modules['%d' % i].conv2.weight])
    bn_tmp = [enc_modules.layer2._modules['%d' % i].bn2.weight,
              enc_modules.layer2._modules['%d' % i].bn2.bias,
              enc_modules.layer2._modules['%d' % i].bn2.running_mean,
              enc_modules.layer2._modules['%d' % i].bn2.running_var]
    enc_resblock_weights.append(bn_tmp)

    if i == 0:
        enc_resblock_weights.append([enc_modules.layer2._modules['%d' % i].downsample._modules['0'].weight])
        bn_tmp = [enc_modules.layer2._modules['%d' % i].downsample._modules['1'].weight,
                  enc_modules.layer2._modules['%d' % i].downsample._modules['1'].bias,
                  enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_mean,
                  enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_var]
        enc_resblock_weights.append(bn_tmp)

# layer3
for i in range(2):
    enc_resblock_weights.append([enc_modules.layer3._modules['%d' % i].conv1.weight])
    bn_tmp = [enc_modules.layer3._modules['%d' % i].bn1.weight,
              enc_modules.layer3._modules['%d' % i].bn1.bias,
              enc_modules.layer3._modules['%d' % i].bn1.running_mean,
              enc_modules.layer3._modules['%d' % i].bn1.running_var]
    enc_resblock_weights.append(bn_tmp)

    enc_resblock_weights.append([enc_modules.layer3._modules['%d' % i].conv2.weight])
    bn_tmp = [enc_modules.layer3._modules['%d' % i].bn2.weight,
              enc_modules.layer3._modules['%d' % i].bn2.bias,
              enc_modules.layer3._modules['%d' % i].bn2.running_mean,
              enc_modules.layer3._modules['%d' % i].bn2.running_var]
    enc_resblock_weights.append(bn_tmp)

    if i == 0:
        enc_resblock_weights.append([enc_modules.layer3._modules['%d' % i].downsample._modules['0'].weight])
        bn_tmp = [enc_modules.layer3._modules['%d' % i].downsample._modules['1'].weight,
                  enc_modules.layer3._modules['%d' % i].downsample._modules['1'].bias,
                  enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_mean,
                  enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_var]
        enc_resblock_weights.append(bn_tmp)

# layer4
for i in range(2):
    enc_resblock_weights.append([enc_modules.layer4._modules['%d' % i].conv1.weight])
    bn_tmp = [enc_modules.layer4._modules['%d' % i].bn1.weight,
              enc_modules.layer4._modules['%d' % i].bn1.bias,
              enc_modules.layer4._modules['%d' % i].bn1.running_mean,
              enc_modules.layer4._modules['%d' % i].bn1.running_var]
    enc_resblock_weights.append(bn_tmp)

    enc_resblock_weights.append([enc_modules.layer4._modules['%d' % i].conv2.weight])
    bn_tmp = [enc_modules.layer4._modules['%d' % i].bn2.weight,
              enc_modules.layer4._modules['%d' % i].bn2.bias,
              enc_modules.layer4._modules['%d' % i].bn2.running_mean,
              enc_modules.layer4._modules['%d' % i].bn2.running_var]
    enc_resblock_weights.append(bn_tmp)

    if i == 0:
        enc_resblock_weights.append([enc_modules.layer4._modules['%d' % i].downsample._modules['0'].weight])
        bn_tmp = [enc_modules.layer4._modules['%d' % i].downsample._modules['1'].weight,
                  enc_modules.layer4._modules['%d' % i].downsample._modules['1'].bias,
                  enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_mean,
                  enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_var]
        enc_resblock_weights.append(bn_tmp)
outputs = []
enc_weights.extend(enc_resblock_weights)
print("transform weights to numpy, transpose, and save ... ")
for i, layer in enumerate(enc_weights):
    if len(layer) == 1:  # Conv
        w = to_numpy(layer[0])
        layer[0] = np.transpose(w, [2, 3, 1, 0])

    if len(layer) == 4:  # BN
        layer = [to_numpy(l) for l in layer]

    for i in range(len(layer)):
        print(layer[i].shape)
    print("==========")
    outputs.append(layer)
with open('models/enc_weights_layerwise.pkl', 'wb') as df:
    pickle.dump(outputs, df)
exit("Done saving")
"""

"""
enc_weights = []
enc_modules = encoder._modules['encoder']
enc_weights.append(enc_modules.conv1.weight)
enc_weights.append(enc_modules.bn1.weight)
enc_weights.append(enc_modules.bn1.bias)
enc_weights.append(enc_modules.bn1.running_mean)
enc_weights.append(enc_modules.bn1.running_var)
enc_resblock_weights = []
# layer1
for i in range(2):
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].conv1.weight)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.weight)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.bias)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.running_mean)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn1.running_var)

    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].conv2.weight)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.weight)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.bias)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.running_mean)
    enc_resblock_weights.append(enc_modules.layer1._modules['%d' % i].bn2.running_var)

# layer2
for i in range(2):
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].conv1.weight)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.weight)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.bias)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.running_mean)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn1.running_var)

    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].conv2.weight)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.weight)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.bias)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.running_mean)
    enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].bn2.running_var)

    if i == 0:
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['0'].weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].weight)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].bias)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_mean)
        enc_resblock_weights.append(enc_modules.layer2._modules['%d' % i].downsample._modules['1'].running_var)

# layer3
for i in range(2):
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].conv1.weight)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.weight)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.bias)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.running_mean)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn1.running_var)

    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].conv2.weight)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.weight)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.bias)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.running_mean)
    enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].bn2.running_var)

    if i == 0:
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['0'].weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].weight)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].bias)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_mean)
        enc_resblock_weights.append(enc_modules.layer3._modules['%d' % i].downsample._modules['1'].running_var)

# layer4
for i in range(2):
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].conv1.weight)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.weight)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.bias)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.running_mean)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn1.running_var)

    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].conv2.weight)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.weight)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.bias)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.running_mean)
    enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].bn2.running_var)

    if i == 0:
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['0'].weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].weight)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].bias)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_mean)
        enc_resblock_weights.append(enc_modules.layer4._modules['%d' % i].downsample._modules['1'].running_var)
"""


def save_features(features):
    """
    features: List of torch.Tensor
        will be converted to List of numpy.ndarray, and saved
    """
    output = [to_numpy(f) for f in features]
    with open('raw_features.pkl', 'wb') as df:
        pickle.dump(output, df)