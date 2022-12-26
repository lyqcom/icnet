"""Evaluation Metrics for Semantic Segmentation"""
# import torch
import numpy as np
import mindspore as ms
import mindspore.ops as ops

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    #计算像素准确度和平均交并比
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, pred, label):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        # def evaluate_worker(self, pred, label):
        correct, labeled = batch_pix_accuracy(pred, label)
        inter, union = batch_intersection_union(pred, label, self.nclass)

        sum = ms.ops.ReduceSum()

        self.total_correct += correct
        self.total_label += labeled
        # if self.total_inter.device != inter.device:
        #     self.total_inter = self.total_inter.to(inter.device)
        #     self.total_union = self.total_union.to(union.device)
        self.total_inter += inter
        self.total_union += union

        # if isinstance(preds, ms.Tensor):
        #     evaluate_worker(self, preds, labels)
        # elif isinstance(preds, (list, tuple)):
        # for (pred, label) in zip(preds, labels):
        # evaluate_worker(self, preds, labels)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        mean = ops.ReduceMean(keep_dims=False)
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        # print("IoU:", IoU)
        mIoU = mean(IoU, axis=0)
        # print("mIou:", mIoU)
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        zeros = ms.ops.Zeros()
        self.total_inter = zeros(self.nclass, ms.float32)
        self.total_union = zeros(self.nclass, ms.float32)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    # print(output.shape)
    # print(target.shape)
    # output = np.array(output)
    # target = np.array(target)
    # global pixel_correct
    # print(output)
    predict = ms.ops.Argmax(output_type=ms.int32, axis=1)(output) + 1  # 索引加一等于类别
    # （1，19， 1024，2048）-->(1, 1024,2048)
    target = target + 1  # 索引加一等于类别
    # print("predeict", predict)
    # print("tar", target)
    # print("tarshape", target.shape)

    typetrue = ms.float32
    cast = ops.Cast()
    sumtarget = ms.ops.ReduceSum()
    sumcorrect = ms.ops.ReduceSum()
    # print("像素点总和")
    # print(cast(target > 0, typetrue))
    labeled = cast(target > 0, typetrue)
    pixel_labeled = sumtarget(labeled)  # 忽略0的像素点总和
    # print("sum", pixel_labeled)
    try:
        # pixel_correct = ms.ops.ReduceSum((predict == target) * (target > 0)).item()
        pixel_correct = sumcorrect(cast(predict == target, typetrue) * cast(target > 0, typetrue))  # 标记正确的像素和
    except:
        print("predict size: {}, target size: {}, ".format(predict.size(), target.size()))
    # assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    predict = ms.ops.Argmax(output_type=ms.int32, axis=1)(output) + 1  # [N,H,W]
    target = target.astype(ms.float32) + 1  # [N,H,W]

    typetrue = ms.float32
    cast = ops.Cast()
    predict = cast(predict, typetrue) * cast(target > 0, typetrue)
    intersection = cast(predict, typetrue) * cast(predict == target, typetrue)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.

    range = ms.Tensor([0.0, 20.0], ms.float32)
    hist = ops.HistogramFixedWidth(nclass+1)
    area_inter = hist(intersection, range)
    area_pred = hist(predict, range)
    area_lab = hist(target, range)
    # print(area_inter)
    # print(area_pred)
    # print(area_lab)
    area_union = area_pred + area_lab - area_inter
    # print(area_union)
    area_inter = area_inter[1:]
    area_union = area_union[1:]
    sum = ms.ops.ReduceSum()
    assert sum(cast(area_inter > area_union, typetrue)) == 0, "Intersection area should be smaller than Union area"
    return cast(area_inter, typetrue), cast(area_union, typetrue)


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return area_intersection, area_union


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

