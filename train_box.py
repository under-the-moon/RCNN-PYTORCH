"""
@Time ：2022/7/5 9:08
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""

"""
Paper nots:
    Each function d⋆(P) (where ⋆ is one of x,y,h,w) is modeled as a linear function of the pool5 features of proposal P ,
    denoted by φ5 (P ). (The dependence of φ5 (P ) on the image data is implicitly assumed.) Thus we have d⋆(P) = wT⋆φ5(P),
     where w⋆ is a vector of learnable model parameters. We learn w⋆ by optimizing the regularized least squares objective
     (ridge regression)
"""

import os
import joblib
from sklearn.linear_model import Ridge


def train_bbox(features, deltas, work_dir):
    clf = Ridge(alpha=0.1)
    clf.fit(features, deltas)
    # save ridge moel
    joblib.dump(clf, os.path.join(work_dir, 'ridge.pkl'))
    return clf
