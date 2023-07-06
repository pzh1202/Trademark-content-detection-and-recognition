import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 针对2007年VOC，使用的11个点计算AP，现在不使用
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  # [0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
        mpre = np.concatenate(([0.], prec, [0.]))  # [0.  1.,     0.6666, 0.4285, 0.3043,  0.]

        # compute the precision envelope
        # 计算出precision的各个断点(折线点)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # [1.     1.     0.6666 0.4285 0.3043 0.    ]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # precision前后两个值不一样的点
        print(mrec[1:], mrec[:-1])
        print(i)  # [0, 1, 3, 4, 5]

        # AP= AP1 + AP2+ AP3+ AP4
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':
    from numpy import *
    '''
    recall = [0.11547344110854503, 0.2807017543859649, 0.3047619047619048, 0.3221476510067114, 0.3356643356643357, 0.35036496350364965, 0.3667953667953668, 0.3861788617886179, 0.4166666666666667, 0.4567307692307692, 0.0]
    precision = [1.0, 0.96, 0.96, 0.96, 0.96, 0.96, 0.95, 0.95, 0.95, 0.95, 0.0]

    ap = voc_ap(recall, precision)
    print(ap)
    '''
    map = [0.4126426769976348, 0.6617540149167418, 0.4676112434291867, 0.41685637814670073, 0.4720406914278499, 0.6192705160437718, 0.5688126003517089, 0.8280328020720022, 0.5961108831807629, 0.40052565547662217, 0.7658730158730158, 0.7171610799153455, 0.3811484957722651, 0.4161046438626239, 0.7145833333333333, 0.4341340843860101, 0.39281608541551516, 0.3481222331844171, 0.20200364223240858, 0.9371614301191765, 0.34368794707138145, 0.43708787472231725, 0.7143404299365036, 0.3191304769877399]
    print(mean(map))
