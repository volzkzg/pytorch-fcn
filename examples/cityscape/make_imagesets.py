import os
import os.path as osp

root = osp.expanduser('~/data/datasets')
for split in ['train', 'val', 'test']:
    dataset_dir = osp.join(root, 'cityscapes/')

    img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
    lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)

    for city in os.listdir(img_dir):
        city_img_dir = osp.join(img_dir, city)
        if not osp.isdir(city_img_dir):
            continue
        with open(osp.join(city_img_dir, 'imgsets.txt'), 'w') as imgsets_file:
            for filename in os.listdir(city_img_dir):
                if '8bit.png' not in filename:
                    continue
                name, _ = filename.split('_left')
                imgsets_file.write(name + '\n')

    for city in os.listdir(lbl_dir):
        city_lbl_dir = osp.join(lbl_dir, city)
        if not osp.isdir(city_lbl_dir):
            continue
        with open(osp.join(city_lbl_dir, 'imgsets.txt'), 'w') as imgsets_file:
            for filename in os.listdir(city_lbl_dir):
                if 'color.png' not in filename:
                    continue
                name, _ = filename.split('_gt')
                imgsets_file.write(name + '\n')
