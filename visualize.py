import os
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.ops as ops
from models import ICNet
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset.vision.py_transforms as transforms

__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',
           'show_prediction', 'show_colorful_images', 'save_colorful_images']


def _img_transform(image):
    # image_transform = tc.Compose([  # 组成一个变换列表
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # print("img", image)
    # # print("len", len(image))
    #
    # image = image_transform(image)
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    image = totensor(image)
    image = normalize(image)
    return image


def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    if show_no_back:
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% || mean_pixel_acc: %.3f%%' % (
            mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))
    else:
        lines.append('mean_IU: %.3f%% || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))
    lines.append('=================================================')
    line = "\n".join(lines)

    print(line)


def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):
        if i != background:
            img[np.where(label == i)] = colors[i]
    if show255:
        img[np.where(label == 255)] = 255

    return img


def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)
    set_img_color(im, pred, colors, background)
    out = np.array(im)

    return out


def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    im.show()


def save_colorful_images(prediction, filename, output_dir, palettes):
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])
    fn = os.path.join(output_dir, filename)
    out_dir = os.path.split(fn)[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    im.save(fn)


def get_color_pallete(npimg, dataset='pascal_voc'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    if dataset in ('pascal_voc', 'pascal_aug'):
        npimg[npimg == -1] = 255
    # put colormap
    if dataset == 'ade20k':
        npimg = npimg + 1
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(adepallete)
        return out_img
    elif dataset == 'citys':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityspallete)
        return out_img
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)

adepallete = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
    5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
    143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,
    7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,
    10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,
    20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,
    20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,
    31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,
    0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,
    0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,
    31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,
    194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
    0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,
    0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
    163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
    10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,
    255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
    133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

def _class_to_index(mask):
    # assert the value
    values = np.unique(mask)
    _key = np.array([-1, -1, -1, -1, -1, -1,
                     -1, -1, 0, 1, -1, -1,
                     2, 3, 4, -1, -1, -1,
                     5, -1, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15,
                     -1, -1, 16, 17, 18])
    _mapping = np.array(range(-1, len(_key) - 1)).astype('int32')
    for value in values:
        assert (value in _mapping)
    # 获取mask中各像素值对应于_mapping的索引
    index = np.digitize(mask.ravel(), _mapping, right=True)
    # 依据上述索引index，根据_key，得到对应的mask图
    return _key[index].reshape(mask.shape)

def _mask_transform(mask):
    mask = _class_to_index(np.array(mask).astype('int32'))
    return np.array(mask).astype('int32')

if __name__ == '__main__':
    model = ICNet(nclass=19, backbone='resnet50', istraining=False)
    ckpt_file_name = '/root/bigpingping/ckpt3/ICNet_3-100_106.ckpt'
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(model, param_dict)
    image_path = 'Test/train_bochum_000000_000600_leftImg8bit.png'
    image = Image.open(image_path).convert('RGB')
    image = _img_transform(image)
    image = ms.Tensor(image)

    # print("image", image.shape)
    expand_dims = ops.ExpandDims()
    image = expand_dims(image, 0)

    squeeze = ops.Squeeze()
    outputs = model(image, image)
    pred = ms.ops.Argmax(axis=1)(outputs[0])
    # print(pred.shape)
    pred = pred.asnumpy()
    pred = pred.squeeze(0)
    pred = get_color_pallete(pred, "citys")
    pred.save('Test/visual_pred_random.png')
