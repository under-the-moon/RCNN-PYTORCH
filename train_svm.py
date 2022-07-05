"""
@Time ：2022/7/5 8:31
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""
import os
import numpy as np
from sklearn.svm import SVC
import joblib


def train_svm(feats, y, num_classes, work_dir):
    clf = SVC(probability=True)
    clf.fit(feats, y)
    joblib.dump(clf, os.path.join(work_dir, 'svm.pkl'))

# hard negative mining
# def train_hnm_svm(feats, y, num_classes, work_dir):
#     feats_ = feats[y != num_classes]
#     y_ = y[y != num_classes]
#     feats_hard = feats[y == num_classes]
#
#     feats_ = np.concatenate([feats_, feats_hard[np.random.randint(0, feats_hard.shape[0], 10)]], axis=0)
#     y_ = np.concatenate([y_, np.ones((10,)) * num_classes], axis=0)
#
#     pred_last = -1
#     pred_now = 0.
#     while pred_now > pred_last:
#         if feats_hard.shape[0] < 1:
#             break
#         clf = SVC(probability=True)
#         clf.fit(feats_, y_)
#         pred = clf.predict(feats_hard)
#
#         pred_prob = clf.predict_proba(feats_hard)
#
#         feats_new_hard = feats_hard[pred != num_classes]
#         num_new_hard = feats_new_hard.shape[0]
#         if num_new_hard < 1:
#             break
#         y_new_hard = pred_prob[pred != num_classes][:, num_classes]
#         pred_last = pred_now
#         count = pred[pred == num_classes].shape[0]
#         pred_now = count / num_new_hard
#
#         # idx = np.argsort(y_new_hard)[::-1][0:-10]
#         idx = np.argsort(y_new_hard)[::-1]
#         y_new_hard_label = np.ones_like(y_new_hard) * num_classes
#         y_ = np.concatenate([y_, y_new_hard_label], axis=0)
#         feats_list = feats_.tolist()
#         for id in idx:
#             feats_list.append(feats_new_hard[id])
#         feats_new_hard_list = []
#         for index in range(num_new_hard):
#             if index in idx:
#                 continue
#             feats_new_hard_list.append(feats_new_hard[index])
#         feats_ = np.array(feats_list)
#         feats_hard = np.array(feats_new_hard_list)
#     joblib.dump(clf, os.path.join(work_dir, 'svm.pkl'))
