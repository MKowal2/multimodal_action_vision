import numpy as np
#from pycocotools import mask
from PIL import Image, ImagePalette # For indexed images
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

colormap = [
    (210, 0, 200),
    (90, 200, 255),
    (0, 199, 0),
    (90, 240, 0),
    (140, 140, 140),
    (100, 60 ,100),
    (255, 255, 0),
    (200, 200 ,0),
    (255, 130, 0),
    (80, 80 ,80),
    (160, 60 ,60),
    (255, 127, 80),
    (0, 139 ,139),
    (200, 250 ,200),
    (0, 128 ,0),
    (127, 255, 212),
    (128, 0 ,128),
    (240, 230, 140),
    (72, 61 ,139),
    (0, 191 ,255),
    (255, 250, 205),
    (230, 230, 250),
    (205, 92 ,92),
    (233, 150, 122),
    (153, 50 ,204),
    (160, 82 ,45),
    (219, 112, 147),
    (245, 222, 179),
    (218, 165, 32),
    (255, 255, 240),
    (178, 34 ,34),
    (210, 105, 30),
    (95, 158 ,160),
    (255, 248, 220),
    (173, 255, 47),
    (224, 255, 255),
    (220, 20 ,60),
    (255, 255, 26),
    (255, 215, 0),
    (255, 140, 0),
    (60, 179, 113),
    (135, 206, 235),
    (100, 149, 237),
    (248, 248, 255),
    (102, 51, 153),
    (164, 89, 58),
    (220, 173, 116),
    (0, 0, 139),
    (255, 182, 193),
    (255, 239, 213),
    (152, 251, 152),
    (47, 79, 79),
    (85, 107, 47),
    (25, 25, 112),
    (128, 0, 0),
    (0, 255, 255),
    (238, 130, 238),
    (147, 112, 219),
    (143, 188, 139),
    (102, 0, 102),
    (69, 33, 84),
    (50, 205, 50),
    (255, 105, 180),
   ]

colors_old = [
    [210, 0, 200],
    [90, 200, 255],
    [0, 199, 0],
    [90, 240, 0],
    [140, 140, 140],
    [100, 60,100],
    [255, 255, 0],
    [200, 200,0],
    [255, 130, 0],
    [80, 80, 80],
    [160, 60,60],
    [255, 127, 80],
    [0, 139,139],
    [200, 250,200],
    [0, 128 ,0],
    [127, 255, 212],
    [128, 0 ,128],
    [240, 230, 140],
    [72, 61 ,139],
    [0, 191 ,255],
    [255, 250, 205],
    [230, 230, 250],
    [205, 92 ,92],
    [233, 150, 122],
    [153, 50 ,204],
    [160, 82 ,45],
    [219, 112, 147],
    [245, 222, 179],
    [218, 165, 32],
    [255, 255, 240],
    [178, 34 ,34],
    [210, 105, 30],
    [95, 158 ,160],
    [255, 248, 220],
    [173, 255, 47],
    [224, 255, 255],
    [220, 20 ,60],
    [255, 255, 26],
    [255, 215, 0],
    [255, 140, 0],
    [60, 179, 113],
    [135, 206, 235],
    [100, 149, 237],
    [248, 248, 255],
    [102, 51, 153],
    [164, 89, 58],
    [220, 173, 116],
    [0, 0, 139],
    [255, 182, 193],
    [255, 239, 213],
    [152, 251, 152],
    [47, 79, 79],
    [85, 107, 47],
    [25, 25, 112],
    [128, 0, 0],
    [0, 255, 255],
    [238, 130, 238],
    [147, 112, 219],
    [143, 188, 139],
    [102, 0, 102],
    [69, 33, 84],
    [50, 205, 50],
    [255, 105, 180]
   ]



def open_pil(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


def convert_rgb_to_class(im, dir, colors):
    out = (np.ones(im.shape[:2]) * 255).astype(np.uint8)
    class_num = 0
    for (label, rgb) in enumerate(colors):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = class_num
        class_num += 1

    # out2 = np.where(out == 255, 1, 0)
    # print(im[np.nonzero(out2)[0],np.nonzero(out2)[1]] )
    assert (out != 255).all(), "rounding errors or missing classes in phav_colors"
    return out.astype(np.uint8)



data_dir_path = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos'


def get_colormap_list():
    file = open("colors_classidx.txt", "r")
    color_list = []
    # Repeat for each song in the text file
    count = 0
    for line in file:
        fields = line.split(' ')
        if count > 0:
            color_list.append([int(fields[1]), int(fields[2]), int(fields[3])])
        count += 1
    return color_list

colors = get_colormap_list()

file = open('../data/phav/videos/phavTrainTestlist/norainfog/data_list_uptojump.txt', 'r')

for line in file:
    directory = line.split(' ')
    print(directory[0])
    dir_path = os.path.join(data_dir_path, directory[0])
    n_frame_path = os.path.join(dir_path, 'n_frames')
    f = open(n_frame_path, "r")
    n_frames = int(f.readline(4))
    for i in range(n_frames):
        image_path = os.path.join(dir_path, 'classgt_{:05d}.png'.format(i))
        if os.path.exists(image_path):
            rgb_img = open_pil(image_path)
            img = convert_rgb_to_class(rgb_img, dir_path, colors)
            img = Image.fromarray(img)
            img.save(dir_path + '/proxy_classgt{:05d}.png'.format(i))

# for root, dirs, _ in os.walk(data_dir_path):
#     # open segmentation image in pil
#     n_frame_path = os.path.join(root, 'n_frames')
#     print(root)
#     if os.path.exists(n_frame_path):
#         f = open(n_frame_path, "r")
#         f1 = f.readline(4)
#         n_frames = int(f1)
#         for i in range(n_frames):
#             image_path = os.path.join(root, 'classgt_{:05d}.png'.format(i))
#             if os.path.exists(image_path):
#                 rgb_img = open_pil(image_path)
#                 img = convert_rgb_to_class(rgb_img, root, colors)
#                 img = Image.fromarray(img)
#                 img.save(root +'/proxy_classgt{:05d}.png'.format(i))

        # convert to proper pixel values using colormap
        # save new image with different name