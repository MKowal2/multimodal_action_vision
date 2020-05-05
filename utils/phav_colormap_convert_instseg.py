import numpy as np
#from pycocotools import mask
from PIL import Image, ImagePalette # For indexed images
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def open_pil(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

def convert_rgb_to_class(im, colors):
    out = (np.ones(im.shape[:2]) * 255).astype(np.uint8)
    class_num = 0
    for (label, rgb) in enumerate(colors):
        match_pxls = np.where((im == np.asarray(rgb)).sum(-1) == 3)
        out[match_pxls] = class_num
        class_num += 1

    out = np.where(out == 255, 0, out)
    # print(im[np.nonzero(out2)[0],np.nonzero(out2)[1]] )
    assert (out != 255).all(), "rounding errors or missing classes in phav_colors"
    return out.astype(np.uint8)

def get_colormap_list(inst_color):
    color_list = [[0,0,0]]
    # Repeat for each song in the text file
    for line in inst_color:
        fields = line.split(' ')
        color_list.append([int(fields[1]), int(fields[2]), int(fields[3])])
    return color_list

data_dir_path = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos'
file = open('../data/phav/videos/phavTrainTestlist/norainfog/data_list_full.txt', 'r')

for line in file:
    directory = line.split(' ')
    dir_path = os.path.join(data_dir_path, directory[0])
    print(dir_path)
    n_frame_path = os.path.join(dir_path, 'n_frames')
    f = open(n_frame_path, "r")
    n_frames = int(f.readline(4))
    inst_color_path = os.path.join(dir_path, 'instances.txt')
    g = open(inst_color_path, "r")
    inst_color = [line.rstrip('\n') for line in g][1:]
    color_list = get_colormap_list(inst_color)
    for i in range(n_frames):
        image_path = os.path.join(dir_path, 'instancegt_{:05d}.png'.format(i))
        rgb_img = open_pil(image_path)
        img = convert_rgb_to_class(rgb_img, color_list)
        img = Image.fromarray(img)
        img.save(dir_path + '/idx_instancegt_{:05d}.png'.format(i))