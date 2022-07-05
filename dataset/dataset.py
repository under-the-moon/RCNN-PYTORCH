"""
@Time ：2022/7/4 20:53
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import cv2
import glob
import torch
import os
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rpn.selective_search import selective_search
from utils.box import box_overlap, box_transform


class MyDataset(Dataset):
    def __init__(self, data_path, classes=None,
                 finetune=True, threshold=.3, finetune_threshold=.5,
                 num_positives=32, num_negatives=96):
        self.image_paths = self._parse_data(data_path)
        self.finetune = finetune
        self.threshold = threshold
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.batch_size = num_positives + num_negatives
        self.finetune_threshold = finetune_threshold
        if classes is None:
            classes = ['airplanes', 'ant']
        self.classes = classes

    def _parse_data(self, data_path):
        image_paths = glob.glob(os.path.join(data_path, 'JPEGImages/*/*'))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def _cls(self, image_path):
        for index, cls in enumerate(self.classes):
            if cls in image_path:
                return index

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        cls = self._cls(image_path)
        annotation_path = image_path.replace('JPEGImages', 'Annotations'). \
            replace('image_', 'annotation_').replace('.jpg', '.mat')

        annot = sio.loadmat(annotation_path)['box_coord']
        # x1, y1, x2, y2
        boxes = np.stack([annot[:, 2], annot[:, 0], annot[:, 3], annot[:, 1], [cls] * annot.shape[0]], axis=-1)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images, labels, deltas = self._get_label(image, boxes)
        return images, labels, deltas

    def _get_label(self, image, boxes):
        _, regions = selective_search(image, scale=1, sigma=0.9, min_size=20)
        rects = np.asarray([list(region['rect']) for region in regions])
        # process region proposal
        filter_rects = []
        filter_images = []
        for rect in rects:
            x1, y1, x2, y2 = list(map(int, rect))
            if (x2 - x1) * (y2 - y1) < 50:
                continue
            crop_img = image[y1:y2, x1:x2, :]
            filter_images.append(crop_img)
            filter_rects.append([x1, y1, x2, y2])

        # add boxes to rect for add positive
        for box in boxes:
            x1, y1, x2, y2 = list(map(int, box[0:4]))
            crop_img = image[y1:y2, x1:x2, :]
            filter_rects.append([x1, y1, x2, y2])
            filter_images.append(crop_img)

        filter_images = np.array(filter_images, dtype=np.object0)
        filter_rects = np.array(filter_rects, dtype=np.float32)

        # cal iou
        overlaps = box_overlap(filter_rects, boxes[:, 0:4])
        # get max box index
        argmax_overlaps = np.argmax(overlaps, axis=1)
        max_overlaps = np.max(overlaps, axis=1)
        # if train for finetune iou eq .5
        # else svm training cls iou eq .3
        threshold = self.finetune_threshold if self.finetune else self.threshold

        labels = np.empty(len(argmax_overlaps))
        labels.fill(len(self.classes))

        if not self.finetune:
            # Paper notes: we take only the ground-truth boxes
            # as positive examples for their respective classes and label
            # proposals with less than 0.3 IoU overlap with all instances
            # of a class as a negative for that class.
            pos_ids = np.where(max_overlaps >= .7)[0]
            # pos_ids = np.where(max_overlaps == 1.)[0]
            neg_ids = np.where(max_overlaps < threshold)[0]
            labels[pos_ids] = boxes[argmax_overlaps[pos_ids], 4]
        else:
            pos_ids = np.where(max_overlaps >= threshold)[0]
            neg_ids = np.where(max_overlaps < threshold)[0]

            if len(pos_ids) > self.num_positives:
                pos_ids = np.random.choice(pos_ids, self.num_positives, replace=False)
            labels[pos_ids] = boxes[argmax_overlaps[pos_ids], 4]
            # get rest negative
        neg_ids = np.random.choice(neg_ids, size=self.batch_size - len(pos_ids), replace=False)
        select_ids = list(pos_ids) + list(neg_ids)
        np.random.shuffle(select_ids)
        labels = labels[select_ids]
        deltas = box_transform(filter_rects[select_ids], boxes[argmax_overlaps[select_ids], 0:4])
        images = filter_images[select_ids]
        return images, labels, deltas


class MyDataLoader:
    def __init__(self, dataset, image_size, pad=16):
        self.dataset = dataset
        self.image_size = image_size
        self.pad = pad

    def _collate_fn(self, batch):
        batch_images = [data[0] for data in batch][0]
        batch_labels = [data[1] for data in batch][0]
        batch_deltas = [data[2] for data in batch][0]
        images = []
        for image in batch_images:
            # we dilate the tight bounding box so that at the warped size there are exactly p pixels of
            # warped image context around the original box (we use p = 16).
            h, w, c = image.shape
            new_image = np.zeros((h + 2 * self.pad, w + 2 * self.pad, c), dtype=np.float32)
            new_image[self.pad: h + self.pad, self.pad: w + self.pad, :] = image
            new_image = cv2.resize(new_image, (self.image_size, self.image_size))
            images.append(new_image)

        images = np.array(images)
        images = images.astype(np.float32)
        images = images / 255.
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.from_numpy(images)
        batch_labels = batch_labels.astype(np.int64)
        batch_labels = torch.from_numpy(batch_labels)
        batch_deltas = batch_deltas.astype(np.float32)
        batch_deltas = torch.from_numpy(batch_deltas)
        return images, batch_labels, batch_deltas

    def __call__(self):
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, collate_fn=self._collate_fn)
        return dataloader


# if __name__ == '__main__':
#     dataset = MyDataset('..\\data')
#     dataloader = MyDataLoader(dataset, 227)()
#     for item in dataloader:
#         X, y, z = item
#         print(X.shape, y.shape, z.shape)
