"""Custom losses."""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore.nn as nn
from utils.losses import SoftmaxCrossEntropyLoss, FocalLoss, DiceLoss
import mindspore.ops as ops
import mindspore as ms
from mindspore import context
# import mindspore.nn as mnn
# from torch.autograd import Variable

# context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

__all__ = ['ICNetLoss']

# TODO: optim function
class ICNetLoss(nn.Cell):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.sparse = True
        self.base_loss = SoftmaxCrossEntropyLoss(num_cls=19, ignore_label=-1)
        self.resize_bilinear = nn.ResizeBilinear()#输入必须为4D


    def construct(self, *inputs):
        preds, target = inputs
        # inputs = tuple(list(preds) + [target])
        # print("len", len(preds))
        # print("target", target.shape)
        pred = preds[0]
        pred_sub4 = preds[1]
        pred_sub8 = preds[2]
        pred_sub16 = preds[3]

        # print("imgshape", pred_sub4.shape)
        # pred4 = ops.Argmax(axis=1)(pred_sub4)
        # pred8 = ops.Argmax(axis=1)(pred_sub8)
        # pred16 = ops.Argmax(axis=1)(pred_sub16)

        # print(pred_sub4.shape)
        # [batch, H, W] -> [batch, 1, H, W]
        expand_dims = ops.ExpandDims()
        if target.shape[0] == 720 or target.shape[0] == 1024:
            target = expand_dims(target, 0).astype(ms.dtype.float32)
            target = expand_dims(target, 0).astype(ms.dtype.float32)
        else:
            target = expand_dims(target, 1).astype(ms.dtype.float32)
        # print("target", target.shape)
        h, w = pred.shape[2:]
        # target = expand_dims(target, 0).astype(ms.dtype.float32)
        #target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        # target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        # target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub4 = self.resize_bilinear(target, size=(h/4, w/4)).squeeze(1)
        # print(int(pred_sub4.shape[2]), int(pred_sub4.shape[3]))
        target_sub8 = self.resize_bilinear(target, size=(h/8, w/8)).squeeze(1)
        # print((int(pred_sub8.shape[2]), int(pred_sub8.shape[3])))
        target_sub16 = self.resize_bilinear(target, size=(h/16, w/16)).squeeze(1)
        # print(int(pred_sub16.shape[2]), int(pred_sub16.shape[3]))

        # pred_sub4 = pred_sub4.reshape(3, 180, 180, 19)
        loss1 = self.base_loss(pred_sub4, target_sub4)
        loss2 = self.base_loss(pred_sub8, target_sub8)
        loss3 = self.base_loss(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

