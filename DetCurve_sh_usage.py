





#-------------------------------------------------------------------------
import scipy.io
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy.stats import norm
from DetCurve_sh import DetCurve

y = scipy.io.loadmat('y.mat')
y=y['y']
y=y.tolist()[0]

ypred = scipy.io.loadmat('ypred.mat')
ypred=ypred['ypred']
ypred=ypred.tolist()[0]

y = np.array(y)
scores = np.array(ypred)

DetCurve(y,scores)  # y: labels (Numdata*1)
#-------------------------------------------------------------------------







"""
#-------------------------------------------------------------------------
# install nihtly bulid version:
# pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn
# conda install scikit-learn
# conda install scipy
# det curve : sklearn
import numpy as np
from sklearn import metrics
from sklearn.metrics import detection_error_tradeoff_curve
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, fnr, thresholds = metrics.detection_error_tradeoff_curve(y_true, y_scores)
#-------------------------------------------------------------------------




#-------------------------------------------------------------------------
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fps, tps, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fns=1-fps
from DETCurve import DETCurve
DETCurve(fpr,fnr)
#-------------------------------------------------------------------------





#-------------------------------------------------------------------------
import scipy.io
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y = scipy.io.loadmat('y.mat')
y=y['y']
y=y.tolist()[0]

ypred = scipy.io.loadmat('ypred.mat')
ypred=ypred['ypred']
ypred=ypred.tolist()[0]

y = np.array(y)
scores = np.array(ypred)
fps, tps, thresholds = metrics.roc_curve(y, scores, pos_label=1)
fns=1-tps

from DETCurve import DETCurve
DETCurve(fps,fns)
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
# conda install bob.measure
# install on py36 environment
negatives=[y[j] for j in range(len(y)) if y[j] == 0]
positives=[y[j] for j in range(len(y)) if y[j] == 1]
from matplotlib import pyplot as plt
curve=bob.measure.det(negatives, positives, n_point)
fps=curve[1,:]
fns=curve[1,:]
plt.plot(fps,fns)
#-------------------------------------------------------------------------

"""










