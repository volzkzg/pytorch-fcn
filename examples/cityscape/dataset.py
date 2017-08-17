import os
import os.path as osp
from PIL import Image

root = osp.expanduser('~/data/datasets')
for split in ['train', 'val', 'test']:
    dataset_dir = osp.join(root, 'cityscapes/')

    img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
    lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)

    for city in os.listdir(img_dir):
        city_img_dir = osp.join(img_dir, city)
        city_lbl_dir = osp.join(lbl_dir, city)
        imgsets_file = osp.join(city_img_dir, 'imgsets.txt')

        if not osp.isdir(city_img_dir):
            continue

        print('In ' + str(city))

        for name in open(imgsets_file):
            name = name.strip()
            img_file = osp.join(city_img_dir, '%s_leftImg8bit.png' % name)
            img = Image.open(img_file)
            w, h = img.size
            img = img.resize((w / 2, h / 2), Image.BILINEAR)
            img.save(osp.join(city_img_dir, '%s.png' % name), "PNG")

        for name in open(imgsets_file):
            name = name.strip()
            lbl_file = osp.join(city_lbl_dir, '%s_gtFine_labelIds.png' % name)
            lbl = Image.open(lbl_file)
            w, h = lbl.size
            lbl = lbl.resize((w / 2, h / 2), Image.NEAREST)
            lbl.save(osp.join(city_lbl_dir, '%s.png' % name), "PNG")

        print('OUT ' + str(city))
