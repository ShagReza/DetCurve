


#-------------------------------------------------------------------------
import scipy.io
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy.stats import norm


def DetCurve(y,scores):
    det_fpr, tps, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    det_fnr=1-tps
   
    fig, ax_det = plt.subplots(1,1, figsize=(5, 5))
    ax_det.set_title('Detection Error Tradeoff (DET) curves')
    ax_det.set_xlabel('False Positive Rate')
    ax_det.set_ylabel('False Negative Rate')
    ax_det.set_xlim(-3, 3)
    ax_det.set_ylim(-3, 3)
    ax_det.grid(linestyle='--')
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locs = norm.ppf(ticks)
    tick_lbls = [
        '{:.0%}'.format(s) if (100*s).is_integer() else '{:.1%}'.format(s)
        for s in ticks
    ]
    plt.sca(ax_det)
    plt.xticks(tick_locs, tick_lbls)
    plt.yticks(tick_locs, tick_lbls)
    ax_det.plot(norm.ppf(det_fpr), norm.ppf(det_fnr))
#-------------------------------------------------------------------------




