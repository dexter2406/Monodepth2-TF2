import os
from PIL import Image
from tqdm import tqdm


def getListFiles(path):
    res = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            res.append(os.path.join(root, filespath))
    return res


def print_file_paths(path):
    res = getListFiles(path)
    listSpicalpath = set()
    for i in tqdm(res):
        if os.path.splitext(i)[1] == ".png":
            png2jpg(i)
            os.remove(i)
            listSpicalpath.add(i)
    print(listSpicalpath)
    print(len(listSpicalpath))


def png2jpg(img_path):
    file_path = img_path.split(".")[:-1]
    out_path = ''.join([file_path[0], '.jpg'])
    img = Image.open(img_path)
    try:
        img.save(out_path)
    except:
        print(img_path, 'fails')




if __name__ == '__main__':
    path = r"E:\Datasets\KITTI\KITTI_odom\kitti_odom"
    print_file_paths(path)
