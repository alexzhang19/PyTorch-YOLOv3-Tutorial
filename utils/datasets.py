#!/usr/bin/env python
# coding: utf-8

"""
@File      : dataset.py
@author    : alex
@Date      : 2020/3/23
@Desc      :
"""

import os
import glob
import copy
import torch
import random
import warnings
import numpy as np
from PIL import Image
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torchvision.transforms as transforms

path = os.path


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    if targets is not None:
        targets[:, 2] = 1 - targets[:, 2]
    return images, targets


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


root_dir = r"F:\dataset\SafetyHelmet"


class ListDataSet(Dataset):
    def __init__(self, data_set: str, img_size: int = 416, augment: bool = True,
                 multi_scale: bool = True, normalized_labels: bool = True):
        assert data_set in ["train", "valid"], "data_set mast in ['train', 'valid']."

        items = self.read_lines(path.join(root_dir, "sets", data_set + ".txt"))
        self.img_files = [path.join(root_dir, "images", v) for v in items]

        assert len(self.img_files) != 0, "dir path have no img."
        self.label_files = [path.join(root_dir, "annots", "new_" + v[:-4] + ".xml") for v in items]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multi_scale = multi_scale
        self.total_img = len(self.img_files)
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.names = self.get_names()

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % self.total_img].rstrip()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        targets = None
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if os.path.exists(label_path):
            gts = self.get_gt(label_path)
            if len(gts) == 0:
                targets = None
            else:
                boxes0 = torch.from_numpy(np.array(gts, dtype=np.float32))
                # Extract coordinates for unpadded + unscaled image
                boxes = copy.deepcopy(boxes0)

                x1 = copy.deepcopy(boxes[:, 1])
                y1 = copy.deepcopy(boxes[:, 2])
                x2 = copy.deepcopy(boxes[:, 3])
                y2 = copy.deepcopy(boxes[:, 4])
                w0 = (x2 - x1) / w
                h0 = (y2 - y1) / h

                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]

                # Returns (x, y, w, h),normal
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h
                boxes[:, 3] = w0 * (w_factor / padded_w)
                boxes[:, 4] = h0 * (h_factor / padded_h)
                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes  # batch_id, cls, nx, ny, nw, nh,

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return img_path, img, targets

    def get_names(self):
        name_path = path.join(root_dir, "names.txt")
        if not path.exists(name_path):
            raise ValueError("names file path not exists.")
        return [v.strip() for v in self.read_lines(name_path, is_split=False)]

    def get_gt(self, label_path):
        tree = ET.parse(label_path)
        objs = tree.findall('object')
        num_objs = len(objs)

        rets = []
        # gt_classes = np.zeros((num_objs), dtype=np.int32)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = obj.find('name').text.lower().strip()

            if cls not in self.names:
                continue
            if x1 < 0 or x2 < 0 or x1 > x2 or y1 > y2:
                warnings.warn("%s coord error: x1=%.3f, y1=%.3f, x2=%.3f, y2=%.3f." %
                              (label_path, x1, y1, x2, y2))
                continue
            rets.append([self.names.index(cls), x1, y1, x2, y2])
        return rets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):  # 浅拷贝,结果更新至targets.
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)  # 连接batch标记.

        # Selects new image size every tenth batch
        if self.multi_scale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def read_lines(self, file_path: str, is_split: bool = False, sep: str = None) -> list:
        lines = []
        fp = open(file_path)
        for item in fp.readlines():
            if is_split:
                lines.append(item.strip().split(sep))
            else:
                lines.append(item.strip())
        fp.close()
        lines = [v for v in lines if v]
        return lines

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    # dataset = ListDataSet("train", augment=False, multi_scale=True)
    # num = len(dataset)
    # colors = random_color(11)
    #
    #
    # def data_set_test():
    #     for i in range(num):
    #         img_path, img, targets = dataset[i]
    #         print(i, img_path, img.shape, np.shape(targets))
    #         img = ToNumpy(trans_to_bgr=True)(img)
    #         print("img:", img.shape, np.max(img), np.min(img))
    #
    #         rgns = []
    #         h, w, _ = img.shape
    #         for idx, tar in enumerate(targets):
    #             [_, cls, x0, y0, w0, h0] = tar
    #             rgns.append([x0, y0, w0, h0, cls])
    #         rgns = xywh2xyxy(np.array(rgns), img.shape)
    #         rgns = xyxy2xywh(np.array(rgns), img.shape)
    #         cv2.imwrite(path.join(desktop, "xxx.jpg"), draw_rectangle(img, rgns, colors))
    #
    #
    # def loader_test():
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=4,
    #         shuffle=False,
    #         num_workers=1,
    #         pin_memory=False,
    #         collate_fn=dataset.collate_fn,
    #         drop_last=True
    #     )
    #
    #     for i, data in enumerate(data_loader):
    #         paths, imgs, targets = data
    #         print(i, paths, imgs, targets)
    #         for j in range(2):
    #             print(j, data[0][j], data[1][j].shape, data[2][j])
    #
    #
    # # print("names:", dataset.get_names())
    #
    # # data_set_test()
    # loader_test()
    pass
