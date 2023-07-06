import argparse

import torch
from torchvision import transforms
from model import EAST
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw

import cfg

from preprocess import resize_image
from nms import nms

import matplotlib.pyplot as plt

import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

import os

import paddlehub as hub

from siamesemole.siamese import Siamese

import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)


def detect(img_path, image, image1, model, siamodel, device, pixel_threshold, save_path, numb, quiet=True):
    img = Image.open(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    with torch.no_grad():
        east_detect = model(load_pil(img).to(device))
    y = np.squeeze(east_detect.cpu().numpy(), axis=0)
    y[:3, :, :] = sigmoid(y[:3, :, :])

    print(y.shape)

    cond = np.greater_equal(y[0, :, :], pixel_threshold)
    # plt.imshow(img)

    plt.subplot(2, 2, 1)
    plt.imshow(np.array(y[0, :, :]))
    plt.subplot(2, 2, 2)
    plt.imshow(np.array(y[1, :, :]))
    plt.subplot(2, 2, 3)
    plt.imshow(np.array(y[2, :, :]))
    # plt.subplot(2, 2, 4)
    # plt.imshow(np.array(y[3, :, :]))
    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.show()
    print(cond.shape)
    print('11111111111111111111111111111111111111111')

    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        tex_pre = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[1, i, j] >= cfg.side_vertex_pixel_threshold:
                if y[2, i, j] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[2, i, j] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        im.save(os.path.join(save_path, str(numb) + '_act.jpg'))
        quad_draw = ImageDraw.Draw(quad_im)
        tex_pre_draw = ImageDraw.Draw(tex_pre)
        txt_items = []
        boxes = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):

            if np.amin(score) > 0:

                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                boxes.append(rescaled_geo_list)
                box = rescaled_geo_list

                imge = np.array(Image.open(img_path))

                img_tem = Image.fromarray(
                    imge[int(min(box[1], box[3], box[5], box[7])):int(max(box[1], box[3], box[5], box[7])),
                    int(min(box[0], box[2], box[4], box[6])):int(max(box[0], box[2], box[4], box[6])), :])

                probability1 = siamodel.detect_image(image, img_tem)
                probability2 = siamodel.detect_image(image1, img_tem)

                tex_pre_draw.line([tuple(geo[0]),
                                   tuple(geo[1]),
                                   tuple(geo[2]),
                                   tuple(geo[3]),
                                   tuple(geo[0])], width=2, fill='red')

                if probability1 >= 0.6:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    quad_draw.text(tuple(geo[1]), 'Similarity:%.3f' % probability1, fill=(255, 0, 0), font=None)

                elif probability2 >= 0.6:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    quad_draw.text(tuple(geo[1]), 'Similarity:%.3f' % probability2, fill=(255, 0, 0), font=None)

                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(os.path.join(save_path, str(numb) + '.jpg'))
        tex_pre.save(os.path.join(save_path, str(numb) + '_pre.jpg'))
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)

    return boxes, np.array(Image.open(img_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='./CSVTR/ChineseRetrievalCollection/7天连锁酒店/Google_0004.jpeg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                        default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='images/test.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args


def show_img(imgs: np.ndarray, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure()
        plt.imshow(img, cmap=None if color else 'gray')


def recognition(config, img, model, converter, device):
    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h,
                     interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))

    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))


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


def recog_label(img_path1, img_save_path):
    orientations = 9
    pixels_per_cell = (2, 2)
    cells_per_block = (2, 2)
    threshold = .3
    model_hog = joblib.load('model_name1.npy')

    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    np_images = [cv2.imread(img_path1)]
    # print(img_path1)
    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；

    draw = Image.open(img_path1).copy()

    img_cv1 = cv2.imread(img_path1)
    # img_cv1 = cv2.cvtColor(img_cv1, cv2.COLOR_BGR2RGB)

    for result in results:
        data = result['data']
        save_path = result['save_path']

        for infomation in data:

            text = np.array(infomation['text_box_position'])

            flag_P = 0
            flag_t = 0
            flag_comp = 0
            box1 = text.reshape(8, )

            box = infomation['text_box_position']
            img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(
                max(box[0][1], box[1][1], box[2][1], box[3][1])),
                          int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(
                              max(box[0][0], box[1][0], box[2][0], box[3][0])), :]

            if '经销企业' in infomation['text']:
                img_cv1 = cv2.putText(img_cv1, 'agent:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)

            if '不适宜人群' in infomation['text']:
                img_cv1 = cv2.putText(img_cv1, 'unapplicable:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)

            if '许可证编号' in infomation['text']:
                img_cv1 = cv2.putText(img_cv1, 'QW:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)

            if '执行标准' in infomation['text']:
                img_cv1 = cv2.putText(img_cv1, 'SC:ture', (box1[0], box1[1]), cv2.FONT_HERSHEY_PLAIN,
                                      1, (255, 0, 0), 2)

            img_detect_hog = Image.fromarray(img_detect1)
            img_detect_hog = img_detect_hog.resize((128, 64))
            # print(img.mode)
            gray = img_detect_hog.convert('L')  # convert the image into single channel i.e. RGB to grayscale
            # calculate HOG for positive features
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',
                     feature_vector=True)  # fd= feature descriptor
            fds = fd.reshape(1, -1)  # re shape the image to make a silouhette of hog
            pred = model_hog.predict(
                fds)  # use the SVM model to make a prediction on the HOG features extracted from the window

            if pred == 1:
                if model_hog.decision_function(
                        fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6

                    img_cv1 = cv2.line(img_cv1, (box1[0], box1[1]), (box1[2], box1[3]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[2], box1[3]), (box1[4], box1[5]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[4], box1[5]), (box1[6], box1[7]), (255, 0, 0), 2)
                    img_cv1 = cv2.line(img_cv1, (box1[6], box1[7]), (box1[0], box1[1]), (255, 0, 0), 2)
                    img_cv1 = cv2.putText(img_cv1, 'Similarity:%.3f' % model_hog.decision_function(fds),
                                          (box1[2], box1[3]), cv2.FONT_HERSHEY_PLAIN,
                                          1, (255, 0, 0), 2)

            draw = draw.convert('RGB')

            cv2.imwrite(os.path.join(img_save_path, '12.png'), img_cv1)


if __name__ == '__main__':
    config, args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    model_path = './saved_model/mb3_512_model_epoch_535.pth'

    image = 'img/11.jpg'
    image1 = 'img/144.jpg'
    image2 = 'img/55.jpg'
    # 加载移动端预训练模型
    # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model_reg = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model_reg.load_state_dict(checkpoint['state_dict'])
    else:
        model_reg.load_state_dict(checkpoint)

    model_hog = joblib.load('model_name1.npy')
    # img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂')
    img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/test')
    img_files = sorted([os.path.join(img_paths, img_file) for img_file in os.listdir(img_paths)])
    img_save_path = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/30/')
    img_save_path2 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/29/')
    sigmodel = Siamese()

    conf = [0.78]
    recall = []
    pres = []
    f1 = []
    tyle_img = 'img/test/'
    model_reg = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model_reg.load_state_dict(checkpoint['state_dict'])
    else:
        model_reg.load_state_dict(checkpoint)
    name_paths = ('./CSVTR/数据集')
    name_files = sorted([name_file for name_file in os.listdir(name_paths)])
    # name_files =['注意安全']
    ap = []
    for name in name_files:
        print(name)
        # img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂')
        # img_paths = os.path.abspath('./CSVTR/ChineseRetrievalCollection/test')
        img_paths = os.path.abspath(os.path.join('./CSVTR/数据集/', name))
        img_files = sorted([os.path.join(img_paths, img_file) for img_file in os.listdir(img_paths)])
        print(img_files)

        sigmodel = Siamese()
        image = os.path.join(tyle_img, name + '.jpg')

        numb = 0
        numb_d = 0
        img_save_path = os.path.abspath(os.path.join('./CSVTR/test/', name + '/'))
        print(img_save_path)
        for img_path1 in img_files:
            # print(img_path1)
            # img_path1 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/南京同仁堂/10.jpeg')
            # img_path3 = os.path.abspath('./CSVTR/ChineseRetrievalCollection/img/1.jpeg')
            print(img_path1)
            np_images = [cv2.imread(img_path1)]
            # print(img_path1)
            results = ocr.recognize_text(
                images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                visualization=False,  # 是否将识别结果保存为图片文件；
                box_thresh=0.5,  # 检测文本框置信度的阈值；
                text_thresh=0.5)  # 识别中文文本置信度的阈值；

            draw = Image.open(img_path1).copy()
            img_cv = cv2.imread(img_path1)
            # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            img_cv1 = cv2.imread(img_path1)
            # img_cv1 = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            # imge = np.array(Image.open(img_path1))
            drawim = ImageDraw.Draw(draw)

            or_draw = Image.open(img_path1).copy()
            or_drawim = ImageDraw.Draw(or_draw)

            numb = numb + 1

            for result in results:
                data = result['data']
                save_path = result['save_path']

        ###shengcheng xunlian ji

                for infomation in data:
                    #print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'],
                    #      '\ntext_box_position: ', infomation['text_box_position'])
                    if infomation['text'] == name:
                        numb = numb + 1
                        box = infomation['text_box_position']
                    #    print(box[0][1])
                        img_detect1 = np_images[0][int(min(box[0][1], box[1][1], box[2][1], box[3][1])):int(max(box[0][1], box[1][1], box[2][1], box[3][1])), int(min(box[0][0], box[1][0], box[2][0], box[3][0])):int(max(box[0][0], box[1][0], box[2][0], box[3][0])), :]
                        Img = Image.fromarray(img_detect1)
                        Img.save(os.path.join(img_save_path, str(numb) + '.jpg'))

        #break
