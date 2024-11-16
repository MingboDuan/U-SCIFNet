import numpy as np
import torch
from skimage import measure
from test import cal_tp_pos_fp_neg
from copy import deepcopy

class ROCMetric():

    def __init__(self, nclass, bins):
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    # Input results and labels
    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = -30 + iBin * (255 / self.bins)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)
        return tp_rates, fp_rates, recall, precision, FP

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])

class mIoU():
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct  # number of correctly predicted pixels
        self.total_label += labeled  # number of pixels in the GT (Ground Truth) target
        self.total_inter += inter  # intersection
        self.total_union += union  # union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

class FMeasure():
    def __init__(self, beta=1.0):
        super(FMeasure, self).__init__()
        self.beta = beta
        self.reset()

    def update(self, preds, labels):
        true_positives = ((preds == 1) & (labels == 1)).sum().item()  # True Positive
        false_positives = ((preds == 1) & (labels == 0)).sum().item()  # False Positive
        false_negatives = ((preds == 0) & (labels == 1)).sum().item()  # False Negative

        self.tp += true_positives
        self.fp += false_positives
        self.fn += false_negatives

    def get(self):
        """
        F-Measure, Precision 和 Recall
        return: Precision, Recall, F-Measure
        """
        precision = self.tp / (self.tp + self.fp + np.spacing(1))  # Precision = TP / (TP + FP)
        recall = self.tp / (self.tp + self.fn + np.spacing(1))     # Recall = TP / (TP + FN)
        # F-Measure
        f_measure = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + np.spacing(1))
        return precision, recall, f_measure

    def reset(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives

class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)   # total number of targets
        self.image_area_total = []   # list of predicted regions
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

class IoU():

    def __init__(self):
        super(IoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        inter, union = batch_intersection_union(preds, labels)
        self.total_inter += inter.sum()
        self.total_union += union.sum()

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return IoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0

class nIoU():
    def __init__(self):
        super(nIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        inter, union = batch_intersection_union(preds, labels)
        self.total_inter += inter.sum()
        self.total_union += union.sum()

    def get(self):
        normalized_inter_union = self.total_inter / (np.spacing(1) + self.total_union)
        nIoU = normalized_inter_union
        return nIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0

class Binary_metric():
    'calculate fscore and iou'
    def __init__(self, thr=0.5):
        self.mean_reset()
        self.norm_reset()
        self.thr = thr
        self.cnt = 0

    def mean_reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def norm_reset(self):
        self.norm_metric = {'precision':0., 'recall':0., 'fscore':0., 'iou':0.}
        self.cnt = 0.

    def update(self, pred, target):
        # 确保 pred 和 target 在同一个设备上
        if pred.device != target.device:
            target = target.to(pred.device)

        # for safety
        pred = pred.detach().clone()
        target = target.detach().clone()
        if torch.max(pred) > 1.1:
            pred = torch.sigmoid(pred)
        pred[pred >= self.thr] = 1
        pred[pred < self.thr] = 0
        self.cur_tp = torch.sum((pred == target) * target, dim=(1,2,3))
        self.cur_tn = torch.sum((pred == target) * (1 - target), dim=(1,2,3))
        self.cur_fp = torch.sum((pred != target) * pred, dim=(1,2,3))
        self.cur_fn = torch.sum((pred != target) * (1 - pred), dim=(1,2,3))
        self.tp += self.cur_tp.sum()
        self.tn += self.cur_tn.sum()
        self.fp += self.cur_fp.sum()
        self.fn += self.cur_fn.sum()
        norm_result = self.norm_compute()
        for k in self.norm_metric.keys():
            self.norm_metric[k] += norm_result[k]
        self.cnt += pred.shape[0]

    def get_mean_result(self):
        mean_metric = self.mean_compute()
        self.mean_reset()
        return mean_metric

    def get_norm_result(self):
        for k,v in self.norm_metric.items():
            self.norm_metric[k] /= self.cnt
        norm_metric = deepcopy(self.norm_metric)
        self.norm_reset()
        return norm_metric

    def norm_compute(self):
        eps = 1e-6
        precision = (self.cur_tp / (self.cur_tp + self.cur_fp + eps)).sum()
        recall = (self.cur_tp / (self.cur_tp + self.cur_fn + eps)).sum()
        fscore = (2 * precision * recall / (precision + recall + eps)).sum()
        iou = (self.cur_tp / (self.cur_tp + self.cur_fn + self.cur_fp + eps)).sum()
        return {"precision":precision, "recall":recall, "fscore":fscore, "iou":iou}

    def mean_compute(self):
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        fscore = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fn + self.fp + eps)
        return {"precision":precision, "recall":recall, "fscore":fscore, "iou":iou}
def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()  # 将output 从 True Flase 转成 1 0
    pixel_labeled = (target > 0).float().sum()  # GF中 1的个数
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()  # 预测对的个数
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union