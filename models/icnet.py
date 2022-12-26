"""Image Cascade Network"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .segbase import SegBaseModel
# from torchsummary import summary

import mindspore.nn as nn
import mindspore.ops as ops
# from models.segbase import SegBaseModel
from utils.loss import ICNetLoss
from .base_models.resnet50_v1 import get_resnet50v1b

from dataset import CityscapesDataset
import PIL.Image as Image
import mindspore as ms
from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.dataset.transforms.py_transforms as tc

__all__ = ['ICNet', 'get_icnet', 'get_icnet_resnet50_citys',
           'get_icnet_resnet101_citys', 'get_icnet_resnet152_citys']

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


class ICNet(nn.Cell):
    """Image Cascade Network"""

    def __init__(self, nclass=19, backbone='resnet50', pretrained_base=True, istraining=True):
        super(ICNet, self).__init__()
        self.conv_sub1 = nn.SequentialCell(
            _ConvBNReLU(3, 32, 3, 2),
            _ConvBNReLU(32, 32, 3, 2),
            _ConvBNReLU(32, 64, 3, 2)
        )
        self.istraining = istraining
        self.ppm = PyramidPoolingModule()

        self.backbone = SegBaseModel()

        self.head = _ICHead(nclass)

        self.loss = ICNetLoss()

        self.resize_bilinear = nn.ResizeBilinear()

        self.__setattr__('exclusive', ['conv_sub1', 'head'])

    def construct(self, x, y):
        if x.shape[0] != 1:
            x = x.squeeze()
        # height, width = x.shape[2:]
        # sub 1
        x_sub1 = self.conv_sub1(x)

        h, w = x.shape[2:]
        # sub 2
        x_sub2 = self.resize_bilinear(x, size=(h / 2, w / 2))
        _, x_sub2, _, _ = self.backbone(x_sub2)

        # sub 4
        # x_sub4 = resize_bilinear(x, size=(h/4, w/4))
        _, _, _, x_sub4 = self.backbone(x)
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)
        # yy = y
        outputs = self.head(x_sub1, x_sub2, x_sub4)

        if self.istraining:
            return self.loss(outputs, y)
        else:
            return outputs
        # return outputs


class PyramidPoolingModule(nn.Cell):  # 金字塔池化模块，通过
    def __init__(self, pyramids=None):
        super(PyramidPoolingModule, self).__init__()
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.pool2 = nn.AvgPool2d(kernel_size=12, stride=11)
        self.pool3 = nn.AvgPool2d(kernel_size=7, stride=8)
        self.pool6 = nn.AvgPool2d(kernel_size=3, stride=4)
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, input):
        feat = input
        height, width = input.shape[2:]
        # print("h,w:", height, width)
        # avgpool = ops.ReduceMean(keep_dims=True)
        # for bin_size in self.pyramids:  # 不同尺度的平均池化------待修改
        #     x = avgpool(input, (2, 3))  #
        #     x = resize_bilinear(x, size=(height, width), align_corners=True)
        #     feat = feat + x
        x1 = self.avgpool(input, (2, 3))
        x1 = self.resize_bilinear(x1, size=(height, width), align_corners=True)
        feat = feat + x1
        # print(feat.shape)

        x2 = self.pool2(input)
        x2 = self.resize_bilinear(x2, size=(height, width), align_corners=True)
        feat = feat + x2
        # print(feat.shape)

        x3 = self.pool3(input)
        x3 = self.resize_bilinear(x3, size=(height, width), align_corners=True)
        feat = feat + x3
        # print(feat.shape)

        x6 = self.pool6(input)
        x6 = self.resize_bilinear(x6, size=(height, width), align_corners=True)
        feat = feat + x6
        # print(feat.shape)
        return feat


class _ICHead(nn.Cell):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ICHead, self).__init__()
        # self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_12 = CascadeFeatureFusion12(128, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_24 = CascadeFeatureFusion24(2048, 512, 128, nclass, norm_layer, **kwargs)

        self.conv_cls = nn.Conv2d(128, nclass, 1, has_bias=False)
        self.outputs = list()
        self.resize_bilinear = nn.ResizeBilinear()

    def construct(self, x_sub1, x_sub2, x_sub4):
        outputs = self.outputs
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)

        # x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)

        # print("1", int(x_cff_12.shape[2] * 2), int(x_cff_12.shape[3] * 2))
        h1, w1 = x_cff_12.shape[2:]
        up_x2 = self.resize_bilinear(x_cff_12, size=(h1 * 2, w1 * 2),
                                     align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        h2, w2 = up_x2.shape[2:]
        # print("2", int(up_x2.shape[2] * 2), int(up_x2.shape[3] * 2))
        up_x8 = self.resize_bilinear(up_x2, size=(h2 * 4, w2 * 4),
                                     align_corners=True)  # scale_factor=4,
        outputs.append(up_x8)
        outputs.append(up_x2)
        outputs.append(x_12_cls)
        outputs.append(x_24_cls)
        # 1 -> 1/4 -> 1/8 -> 1/16
        # outputs.reverse()
        # print(up_x8.shape)
        # print(up_x2.shape)
        # print(x_12_cls.shape)
        # print(x_24_cls.shape)
        return outputs


class _ConvBNReLU(nn.Cell):  # 通过
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                              dilation=dilation,
                              group=1, has_bias=False)
        self.bn = norm_layer(out_channels, momentum=0.1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CascadeFeatureFusion12(nn.Cell):  # 通过
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion12, self).__init__()
        self.conv_low = nn.SequentialCell(
            nn.Conv2d(low_channels, out_channels, 3, pad_mode='pad', padding=2, dilation=2, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_high = nn.SequentialCell(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_low_cls = nn.Conv2d(in_channels=out_channels, out_channels=nclass, kernel_size=1, has_bias=False)
        self.resize_bilinear = nn.ResizeBilinear()

        self.scalar_cast = ops.ScalarCast()

        self.relu = ms.nn.ReLU()

    def construct(self, x_low, x_high):
        # resize_bilinear = nn.ResizeBilinear()
        # print(int(x_high.shape[2]), int(x_high.shape[3]))
        h, w = x_high.shape[2:]
        x_low = self.resize_bilinear(x_low, size=(h, w), align_corners=True)
        x_low = self.conv_low(x_low)
        # print(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high

        x = self.relu(x)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class CascadeFeatureFusion24(nn.Cell):  # 通过
    """CFF Unit"""

    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CascadeFeatureFusion24, self).__init__()
        self.conv_low = nn.SequentialCell(
            nn.Conv2d(low_channels, out_channels, 3, pad_mode='pad', padding=2, dilation=2, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_high = nn.SequentialCell(
            nn.Conv2d(high_channels, out_channels, kernel_size=1, has_bias=False),
            norm_layer(out_channels, momentum=0.1)
        )
        self.conv_low_cls = nn.Conv2d(in_channels=out_channels, out_channels=nclass, kernel_size=1, has_bias=False)

        self.resize_bilinear = nn.ResizeBilinear()
        self.relu = ms.nn.ReLU()

    def construct(self, x_low, x_high):
        h, w = x_high.shape[2:]
        # print(int(x_high.shape[2]), int(x_high.shape[3]))
        x_low = self.resize_bilinear(x_low, size=(h, w), align_corners=True)
        x_low = self.conv_low(x_low)
        # print(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high

        x = self.relu(x)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class SegBaseModel(nn.Cell):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass=19, backbone='resnet50', pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = True
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = get_resnet50v1b()

    def construct(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4


if __name__ == '__main__':
    # # #img = torch.randn(1, 3, 256, 256)
    # # model = get_icnet_resnet50_citys()
    # # #outputs = model(img)
    #
    # inputs = torch.randn(1, 3, 720, 720)
    # # with torch.no_grad():
    # #     outputs = model(inputs)
    # # print(len(outputs))		 # 3
    # # print(outputs[0].size()) # torch.Size([1, 19, 200, 200])
    # # print(outputs[1].size()) # torch.Size([1, 19, 100, 100])
    # # print(outputs[2].size()) # torch.Size([1, 19, 50, 50])
    # # print(outputs[3].size()) # torch.Size([1, 19, 50, 50])
    # # import mindspore as ms
    # print("已修改")
    # # shape1 = (3, 2048, 180, 180)
    # # shape2 = (3, 512, 360, 360)
    # # shape3 = (3, 64, 720, 720)
    # # stdnormal = ms.ops.StandardNormal(seed=2)
    # # img1 = stdnormal(shape1)
    # # img2 = stdnormal(shape2)
    # # img3 = stdnormal(shape3)
    # # # print(img.shape)
    # # print('--------------------------------------------------------')
    # # model = _ICHead(nclass=19)
    # # output = model(img3, img2, img1)
    # # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)
    # # pass
    shape = (3, 3, 720, 720)
    # # test2_path = "Test/test2.png"
    # # mask = Image.open(test2_path)
    stdnormal = ms.ops.StandardNormal(seed=2)
    img = stdnormal(shape)
    # # mask =
    # # print(img.shape)
    model = ICNet()
    # # # # # loss = ICNetLoss()
    # # # # # output = module(img, mask)
    # # # # # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)
    # # # # # for m in module.parameters_and_names():
    # # # # # print(output.shape)
    # # # # #     print(m)
    # testdataset = CityscapesDataset()
    # train_ds = ds.GeneratorDataset(testdataset, ["imgs", "masks"], shuffle=False)
    # train_ds = train_ds.batch(batch_size=7)
    # # # # # train_ds = train_ds.map(operations=transforms.ToTensor, input_columns=["imgs"])
    # # # # # # train_ds = create_dataset(train_ds)
    # # # # # # print("datasize", train_ds.get_dataset_size())
    # imgs = []
    # masks = []
    # for data in train_ds.create_dict_iterator():
    #     # print(data["imgs"], data["masks"])
    #     imgs.append(data["imgs"])
    #     masks.append(data["masks"])
    #     break
    # # #     # print(loss)
    # # test1_path = "../Test/bochum_000000_006026_leftImg8bit.png"
    # # test2_path = "../Test/bochum_000000_006026_gtFine_labelIds.png"
    # # img = Image.open(test1_path).convert('RGB')
    # # mask = Image.open(test2_path)
    # # img = ms.Tensor(img)
    # # #     # print("img", img)
    # # #     print("img_shape:", img.shape)
    # # #     print("mask_shape", mask.shape)
    # # #     # print("mask", mask)
    # img = imgs[0]
    # mask = masks[0]
    # print("mask_shape", mask.shape)
    outputs = model(img, False)
    for output in outputs:
        print(output.shape)
    #     # output = model(img)
    #     # print(output[0])
    #     print("-----------------------------------------------------")
    #     break
    # print(loss)
    pass
