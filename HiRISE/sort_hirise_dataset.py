# 1. create directories compatible with keras image datagenerator
# 2. create symbolic link from download images to new directories
# Run this script on the directory where you want to place the dataset.

import os,random

VAL_FRAC = 0.1
random.seed(1024)

LABEL_FILE = os.path.join(os.path.dirname(__file__), 'labels-map-proj-v3.txt')

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map-proj-v3')

CLASSES = ['other', 'crater', 'dark_dune', 'slope_streak', 'bright_dune', 'impact_ejecta', 'swiss_cheese', 'spider']

def make_dir(p):
    if not os.path.exists(p): os.mkdir(p)

def make_data_dir():
    for data_type in ['train', 'val']:
        make_dir(os.path.join(os.path.dirname(__file__), data_type))
        for cls in CLASSES:
            make_dir(os.path.join(os.path.dirname(__file__), data_type, cls))

def link_img():
    #load file
    label_file = LABEL_FILE
    with open(label_file) as f:
        lines = f.readlines()

    val_idx = random.sample(range(len(lines)), int(len(lines)*VAL_FRAC))

    for idx, ln in enumerate(lines):
        img_file, label = ln.split()
        dist = 'val' if idx in val_idx else 'train'
        cmd_str = 'ln -s {} {}'.format(os.path.join(IMG_DIR, img_file), os.path.join(os.path.dirname(os.path.abspath(__file__)), dist, CLASSES[int(label)]))
        print('[{}] {} => {}'.format(dist, img_file, label))
        os.system(cmd_str)

if __name__ == '__main__':
    make_data_dir()
    link_img()
