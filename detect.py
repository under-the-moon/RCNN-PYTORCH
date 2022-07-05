"""
@Time ：2022/7/5 12:39
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import yaml
import numpy as np
from net.model import Model
from rpn.selective_search import selective_search
from utils.box import bbox_transform_inv
from utils.nms import nms

cfg = yaml.safe_load(open('config/rcnn.yaml'))

pad = cfg['pad']
image_size = cfg['image_size']
classes = cfg['classes']
backbone = cfg['backbone']


def get_proposal(image):
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
        h, w, c = crop_img.shape
        new_image = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.float32)
        new_image[pad: h + pad, pad: w + pad, :] = crop_img
        new_image = cv2.resize(new_image, (image_size, image_size))
        filter_images.append(new_image)
        filter_rects.append([x1, y1, x2, y2])
    filter_images = np.array(filter_images)
    filter_images = np.transpose(filter_images, (0, 3, 1, 2))
    filter_images = filter_images / 255.
    filter_rects = np.array(filter_rects)
    return filter_images, filter_rects


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    for rect in regions:
        x1, y1, x2, y2 = list(map(int, rect))
        w = x2 - x1
        h = y2 - y1
        rect = mpatches.Rectangle(
            (x1, y1), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Model(backbone, len(classes))
    model.load_state_dict(torch.load('weights/efficientnet_b0.pth'))
    model.to(device)
    model.eval()

    image_path = 'data/JPEGImages/airplanes/image_0003.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images, rects = get_proposal(image)

    images = torch.from_numpy(images)
    images = images.to(device)

    feats = []
    filter_rects = []
    for index, image in enumerate(images):
        image = torch.unsqueeze(image, dim=0)
        with torch.no_grad():
            out, feat = model(image)
        feat_score = torch.softmax(out, dim=-1)[0]
        score = torch.argmax(feat_score, dim=-1)
        if score == len(classes):
            continue
        feats.append(feat.cpu().numpy())
        filter_rects.append(rects[index])

    feats = np.concatenate(feats, axis=0)
    filter_rects = np.array(filter_rects)

    # load svm model
    svc_model = joblib.load('weights/svm.pkl')
    preds = svc_model.predict(feats)
    preds_probs = svc_model.predict_proba(feats)
    preds_probs = np.max(preds_probs, axis=-1)

    # feats = feats[preds != len(classes)]
    scores = preds_probs[preds != len(classes)]
    filter_rects = filter_rects[preds != len(classes)]
    feats = feats[preds != len(classes)]

    # load box regresion model
    box_model = joblib.load('weights/ridge.pkl')

    pred_deltas = box_model.predict(feats)
    pred_boxes = bbox_transform_inv(filter_rects, pred_deltas)
    keep = nms(pred_boxes, scores)
    pred_boxes = pred_boxes[keep, :]
    scores = scores[keep]

    # show result
    show_rect(image_path, pred_boxes[0:1])
