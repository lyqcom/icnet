# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=19, ignore_label=-1):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss

class FocalLoss(nn.Cell):
    def __init__(self,
                 ignore_label=-1,
                 sparse=True,
                 reduction='mean',
                 batch_average=False):
        super(FocalLoss, self).__init__()

        self.ignore_label = ignore_label
        self.sparse = sparse
        self.reduction = reduction
        self.batch_average = batch_average
        self.cast = P.Cast()
        self.exp = P.Exp()
        self.criterion = SoftmaxCrossEntropyLoss(ignore_label=self.ignore_label)

    def construct(self, logit, label, gamma=2, alpha=0.5):
        n, c, h, w = logit.shape

        logpt1 = -self.criterion(logit, label)

        pt = self.exp(logpt1)
        logpt2 = alpha * logpt1
        focal_loss = -((1 - pt) ** gamma) * logpt2

        if self.batch_average:
             focal_loss /= n
        return focal_loss

class DiceLoss(nn.Cell):
    def __init__(self,
                 num_cls,
                 ignore_label):
        super(DiceLoss, self).__init__()
        self.num_cls = num_cls
        self.one_hot = P.OneHot(axis=1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.softmax = nn.Softmax(axis=1)
        self.dloss = nn.MultiClassDiceLoss(ignore_indiex=ignore_label, activation=self.softmax)

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.dloss(logits, one_hot_labels)
        return loss